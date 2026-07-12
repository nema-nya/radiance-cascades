from setup_context import SetupContext
from frame_context import FrameContext


class SequentialRenderer:
    def __init__(self, rs):
        self.rs = rs

    async def setup(self, setup_ctx: SetupContext):
        for r in self.rs:
            await r.setup(setup_ctx)

    async def draw(self, frame_ctx: FrameContext):
        for r in self.rs:
            await r.draw(frame_ctx)
