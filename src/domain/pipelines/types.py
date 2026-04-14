from dataclasses import dataclass


@dataclass
class AnnotationBox:
    specie: str
    call_type: str
    begin_time: float
    end_time: float
    low_freq: float
    high_freq: float


@dataclass
class PixelBBox:
    class_id: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __iter__(self):
        yield self.class_id
        yield self.x_min
        yield self.y_min
        yield self.x_max
        yield self.y_max


@dataclass
class YoloLabel:
    class_id: int
    xc_rel: float
    yc_rel: float
    w_rel: float
    h_rel: float

    def __iter__(self):
        yield self.class_id
        yield self.xc_rel
        yield self.yc_rel
        yield self.w_rel
        yield self.h_rel
