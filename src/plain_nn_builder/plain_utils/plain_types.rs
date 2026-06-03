use half::f16;

pub trait PlainElement: Clone + Send + Sync {}

impl PlainElement for u8 {}
impl PlainElement for u16 {}
impl PlainElement for u32 {}
impl PlainElement for u64 {}

pub trait PlainValueType: Sized {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

impl PlainValueType for u16 {
    fn to_f32(self) -> f32 {
        f16::from_bits(self).to_f32()
    }

    fn from_f32(value: f32) -> Self {
        f16::from_f32(value).to_bits()
    }
}

impl PlainValueType for u32 {
    fn to_f32(self) -> f32 {
        f32::from_bits(self)
    }

    fn from_f32(value: f32) -> Self {
        value.to_bits()
    }
}
