# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import socket
import struct
import threading
from typing import TYPE_CHECKING, Optional

import msgspec
import uvloop
from msgspec import msgpack

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import get_ip, get_open_port

if TYPE_CHECKING:
    from vllm.v1.engine.core import DPEngineCoreProc

logger = init_logger(__name__)


# All coroutines used as the main of task must apply this decorator.
def _kill_me_if_exception(async_func):

    async def wrapper(*args, **kwargs):
        try:
            # more detail see https://blog.hidva.com/2025/03/26/save-asyncio-task/
            loop = asyncio.get_running_loop()
            if not hasattr(loop, '_vllm_dpcoord_running_task'):
                loop._vllm_dpcoord_running_task = set()  # type: ignore
            running_task: set[
                asyncio.Task] = loop._vllm_dpcoord_running_task  # type: ignore
            task = asyncio.current_task()
            assert task is not None
            running_task.add(task)
            task.add_done_callback(running_task.discard)

            return await async_func(*args, **kwargs)
        except Exception:
            logger.exception("async_func: ex")
            os.abort()

    return wrapper


def _asyncio_loop_main(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()
    return


def _start_asyncio_thread(name):
    loop = uvloop.new_event_loop()
    threading.Thread(target=_asyncio_loop_main,
                     args=(loop, ),
                     name=name,
                     daemon=True).start()
    return loop


class _ParticipantInfo(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]
    dprank: int = 0
    host: str = ''
    port: int = 0


# A simple RPC implementation
#
# Message Format:
#
#  +------+------------+
#  | head |    body    |
#  +------+------------+
#
# head: 4bytes,
# body: based on the value of "head".

# Body:
# +------+
# | step |
# +------+
# step: 4bytes
_START_STEP_REQ = 0x20181218
# No Body
# I Want NoBody NoBody But You~
_START_STEP_RESP = 0x81218102

# Body:
# +-----+-----------------+
# | len | req             |
# +-----+-----------------+
# len: 4bytes, sizeof(req)
# req: encoded _ParticipantInfo
_REG_PART_REQ = 0x20181219
# No Body
# I Want NoBody NoBody But You~
_REQ_PART_RESP = 0x91218102


# @_kill_me_if_exception
async def _start_step_rpc(host: str, port: int, step: int):
    reader, writer = await asyncio.open_connection(host, port)
    reqbuf = struct.pack('=II', _START_STEP_REQ, step)
    writer.write(reqbuf)
    await writer.drain()

    respbuf = await reader.readexactly(4)
    respcode = struct.unpack('=I', respbuf)[0]
    assert respcode == _START_STEP_RESP

    writer.close()
    await writer.wait_closed()
    return


def _start_step_rpc_sync(host: str, port: int, step: int):
    # loop = asyncio.get_event_loop()
    # coro = _start_step_rpc(self._coord_host, self._coord_port,
    # self._local_step)
    # loop.run_until_complete(coro)
    with socket.create_connection((host, port)) as sock:
        reqbuf = struct.pack('=II', _START_STEP_REQ, step)
        sock.sendall(reqbuf)
        respbuf = sock.recv(4)
        respcode = struct.unpack('=I', respbuf)[0]
        assert respcode == _START_STEP_RESP
    return


class _Coordinator:

    def __init__(self, dpsize: int, port: int, loop):
        self._loop = loop
        self._port = port
        self._dpsize = dpsize
        self._step = 0
        self._parts: list[Optional[_ParticipantInfo]] = [
            None for _ in range(dpsize)
        ]
        self._info_decoder = msgpack.Decoder(_ParticipantInfo)

    async def _part_start_step(self, partidx: int):
        pinfo = self._parts[partidx]
        assert pinfo is not None  # may be sleep if pinfo is None

        await _start_step_rpc(pinfo.host, pinfo.port, self._step)
        return

    @_kill_me_if_exception
    async def _do_start_step(self, step: int):
        if step <= self._step:
            return
        self._step = step + 24
        futs = [self._part_start_step(idx) for idx in range(self._dpsize)]
        await asyncio.gather(*futs)
        return

    async def _start_step(self, reader, writer):
        bodybuf = await reader.readexactly(4)
        step = int(struct.unpack('=I', bodybuf)[0])

        if step > self._step:
            self._loop.create_task(self._do_start_step(step))

        resp = struct.pack('=I', _START_STEP_RESP)
        writer.write(resp)
        await writer.drain()
        return

    async def _reg_part(self, reader, writer):
        bodylenbuf = await reader.readexactly(4)
        bodylen, = struct.unpack('=I', bodylenbuf)
        reqbuf = await reader.readexactly(bodylen)
        info: _ParticipantInfo = self._info_decoder.decode(reqbuf)

        assert self._parts[info.dprank] is None
        self._parts[info.dprank] = info

        resp = struct.pack('=I', _REQ_PART_RESP)
        writer.write(resp)
        await writer.drain()
        return

    async def _client_main(self, reader, writer):
        # Create multiple connections for concurrency,
        try:
            while True:
                headbuf = await reader.readexactly(4)
                head, = struct.unpack('=I', headbuf)
                if head == _START_STEP_REQ:
                    await self._start_step(reader, writer)
                    continue
                if head == _REG_PART_REQ:
                    await self._reg_part(reader, writer)
                    continue
                logger.warning("_Coordinator unknown head. head=%s", head)
                break
        except asyncio.IncompleteReadError as ex:
            if ex.partial:  # get non-empty bytes object from readexactly
                logger.exception("client main: ex")
        except Exception:
            logger.exception("client main: ex")
        return

    @_kill_me_if_exception
    async def _main(self):
        server = await asyncio.start_server(self._client_main, '0.0.0.0',
                                            self._port)
        logger.info("Coordinator Rpc Listen addr=0.0.0.0:%s", self._port)
        async with server:
            await server.serve_forever()
        return


def _start_coord(port: int, dpsize: int):
    loop = _start_asyncio_thread("coord")
    coord = _Coordinator(dpsize, port, loop)
    coro = coord._main()
    asyncio.run_coroutine_threadsafe(coro, loop)
    return


class Participant:

    def __init__(self, vllmcfg: VllmConfig, core: "DPEngineCoreProc"):
        # Core Thread
        self._core = core
        self._info_encoder = msgpack.Encoder()
        self._dprank = vllmcfg.parallel_config.data_parallel_rank
        self._coord_port = vllmcfg.parallel_config.get_next_dp_init_port()
        self._coord_host = vllmcfg.parallel_config.data_parallel_master_ip
        dpsize = vllmcfg.parallel_config.data_parallel_size

        # R/W only on Core Thread
        self._local_step = 0
        self._coord_step = 0

        if self._dprank == 0:
            _start_coord(self._coord_port, dpsize)

        port = get_open_port()
        _start_part(self, port)
        return

    def new_step(self):
        # Core Thread
        self._local_step += 1
        if self._local_step <= self._coord_step:
            return

        _start_step_rpc_sync(self._coord_host, self._coord_port,
                             self._local_step)
        return

    def engines_running(self) -> bool:
        # Core Thread
        return self._local_step < self._coord_step

    def core_start_step(self, step):
        # Core Thread
        if step > self._coord_step:
            self._coord_step = step
        return

    async def _start_step(self, reader, writer):
        bodybuf = await reader.readexactly(4)
        step = int(struct.unpack('=I', bodybuf)[0])

        self._core.start_step_threadsafe(step)

        resp = struct.pack('=I', _START_STEP_RESP)
        writer.write(resp)
        await writer.drain()
        return

    async def _client_main(self, reader, writer):
        # Create multiple connections for concurrency,
        try:
            while True:
                headbuf = await reader.readexactly(4)
                head, = struct.unpack('=I', headbuf)
                if head == _START_STEP_REQ:
                    await self._start_step(reader, writer)
                    continue
                logger.warning("_Coordinator unknown head. head=%s", head)
                break
        except asyncio.IncompleteReadError as ex:
            if ex.partial:  # get non-empty bytes object from readexactly
                logger.exception("client main: ex")
        except Exception:
            logger.exception("client main: ex")
        return

    async def _reg_part(self, info: _ParticipantInfo):
        reader, writer = await asyncio.open_connection(self._coord_host,
                                                       self._coord_port)

        infobuf = self._info_encoder.encode(info)
        reqbuf = struct.pack('=II', _REG_PART_REQ, len(infobuf))
        reqbuf += infobuf
        writer.write(reqbuf)
        await writer.drain()

        respbuf = await reader.readexactly(4)
        respcode = struct.unpack('=I', respbuf)[0]
        assert respcode == _REQ_PART_RESP

        writer.close()
        await writer.wait_closed()
        return

    @_kill_me_if_exception
    async def _main(self, port: int):
        server = await asyncio.start_server(self._client_main, '0.0.0.0', port)

        info = _ParticipantInfo(dprank=self._dprank, host=get_ip(), port=port)
        logger.info("Participant Rpc Listen addr=0.0.0.0:%s. RegInfo=%s", port,
                    info)

        while True:
            try:
                await self._reg_part(info)
                break
            except Exception:
                logger.exception('reg part ex')
                await asyncio.sleep(0.1)

        async with server:
            await server.serve_forever()
        return


def _start_part(part: Participant, port: int):
    loop = _start_asyncio_thread("part")
    coro = part._main(port)
    asyncio.run_coroutine_threadsafe(coro, loop)
    return
