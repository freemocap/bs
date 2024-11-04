import time
from typing import Dict
from pydantic import BaseModel, Field

class TimestampMapping(BaseModel):
    """
    A mapping between different timing methods
    """
    camera_timestamps: Dict[int, int]
    utc_time_ns: int = Field(default_factory=time.time_ns, description="UTC time in nanoseconds from `time.time_ns()`")
    perf_counter_ns: int = Field(default_factory=time.perf_counter_ns,
                                 description="Time in nanoseconds from `time.perf_counter_ns()` (arbirtary time base)")
    monotonic_ns: int = Field(default_factory=time.monotonic_ns, description="Time in nanoseconds from `time.monotonic()` (arbitrary time base)")

    # TODO: Pupil uses a C function `get_sys_time_monotonic` - need to verify this matches `time.monotonic`
    def convert_perf_counter_ns_to_unix_ns(self, perf_counter_ns: int) -> int:
        """
        Convert a `time.perf_counter_ns()` timestamp to a unix timestamp
        """
        return self.utc_time_ns + (perf_counter_ns - self.perf_counter_ns)