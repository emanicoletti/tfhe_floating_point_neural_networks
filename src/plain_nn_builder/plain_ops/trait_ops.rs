use crate::plain_nn_builder::plain_ops::*;

pub trait PlainAdd {
    fn add(self, other: Self) -> Self;
}

pub trait PlainMul {
    fn mul(self, other: Self) -> Self;
}

pub trait PlainDiv {
    fn div(self, other: Self) -> Self;
}

pub trait PlainSub {
    fn sub(self, other: Self) -> Self;
}

pub trait PlainTanh where Self: Sized {
    fn tanh(self) -> (Self, Self);
}

impl PlainAdd for u32 {
    fn add(self, other: Self) -> Self {
        add32(self, other)
    }
}

impl PlainMul for u32 {
    fn mul(self, other: Self) -> Self {
        lmul32(self, other)
    }
}

impl PlainDiv for u32 {
    fn div(self, other: Self) -> Self {
        ldiv32(self, other)
    }
}

impl PlainSub for u32 {
    fn sub(self, other: Self) -> Self {
        sub32(self, other)
    }
}

impl PlainTanh for u32 {
    fn tanh(self) -> (Self, Self) {
        tanh32(self)
    }
}

impl PlainAdd for u16 {
    fn add(self, other: Self) -> Self {
        add16(self, other)
    }
}

impl PlainMul for u16 {
    fn mul(self, other: Self) -> Self {
        lmul16(self, other)
    }
}

impl PlainDiv for u16 {
    fn div(self, other: Self) -> Self {
        ldiv16(self, other)
    }
}

impl PlainSub for u16 {
    fn sub(self, other: Self) -> Self {
        sub16(self, other)
    }
}

impl PlainTanh for u16 {
    fn tanh(self) -> (Self, Self) {
        tanh16(self)
    }
}

