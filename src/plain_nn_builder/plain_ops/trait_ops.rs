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

pub trait PlainReLU where Self: Sized {
    fn relu(self) -> (Self, Self);
}

pub trait PlainBackwardReLU where Self: Sized {
    fn backward_relu(self, grad_output: Self) -> Self;
}

pub trait PlainSqrt where Self: Sized {
    fn sqrt(self) -> Self;
}

impl PlainAdd for u32 {
    fn add(self, other: Self) -> Self {
        #[cfg(feature = "exact")]
        { return canonical_add32(self, other); }

        #[cfg(feature = "lmul")]
        { return add32(self, other); }

        #[cfg(feature = "pam")]
        { return add32(self, other); }

        unreachable!("No valid feature selected for add(u32)");
    }
}

impl PlainMul for u32 {
    fn mul(self, other: Self) -> Self {
        #[cfg(feature = "exact")]
        { return canonical_mul32(self, other); }

        #[cfg(feature = "lmul")]
        { return lmul32(self, other); }

        #[cfg(feature = "pam")]
        { return pam_mul32(self, other); }

        unreachable!("No valid feature selected for mul(u32)");
    }
}

impl PlainDiv for u32 {
    fn div(self, other: Self) -> Self {
        #[cfg(feature = "exact")]
        { return canonical_div32(self, other); }

        #[cfg(feature = "lmul")]
        { return ldiv32(self, other); }

        #[cfg(feature = "pam")]
        { return pam_div32(self, other); }

        unreachable!("No valid feature selected for div(u32)");
    }
}

impl PlainSub for u32 {
    fn sub(self, other: Self) -> Self {
        #[cfg(feature = "exact")]
        { return canonical_sub32(self, other); }

        #[cfg(any(feature = "lmul", feature = "pam"))]
        { return sub32(self, other); }

        unreachable!("No valid feature selected for sub(u32)");
    }
}

impl PlainTanh for u32 {
    fn tanh(self) -> (Self, Self) {
        #[cfg(feature = "exact")]
        { return canonical_tanh32(self); }

        #[cfg(feature = "lmul")]
        { return lmul_tanh32(self); }

        #[cfg(feature = "pam")]
        { return pam_tanh32(self); }

        unreachable!("No valid feature selected for tanh(u32)");
    }
}

impl PlainReLU for u32 {
    fn relu(self) -> (Self, Self) {
        relu32(self) // same for all features
    }
}

impl PlainBackwardReLU for u32 {
    fn backward_relu(self, grad_output: Self) -> Self {
        backward_relu32(self, grad_output) // same for all features
    }
}

impl PlainSqrt for u32 {
    fn sqrt(self) -> Self {
        #[cfg(feature = "exact")]
        { return canonical_sqrt32(self); }

        #[cfg(any(feature = "lmul", feature = "pam"))]
        { return sqrt32(self); }

        unreachable!("No valid feature selected for sqrt(u32)");
    }
}

// ==== u16 impls ====

impl PlainAdd for u16 {
    fn add(self, other: Self) -> Self {
        #[cfg(feature = "exact")]
        { return canonical_add16(self, other); }

        #[cfg(any(feature = "lmul", feature = "pam"))]
        { return add16(self, other); }

        unreachable!("No valid feature selected for add(u16)");
    }
}

impl PlainMul for u16 {
    fn mul(self, other: Self) -> Self {
        #[cfg(feature = "exact")]
        { return canonical_mul16(self, other); }

        #[cfg(feature = "lmul")]
        { return lmul16(self, other); }

        #[cfg(feature = "pam")]
        { return pam_mul16(self, other); }

        unreachable!("No valid feature selected for mul(u16)");
    }
}

impl PlainDiv for u16 {
    fn div(self, other: Self) -> Self {
        #[cfg(feature = "exact")]
        { return canonical_div16(self, other); }

        #[cfg(feature = "lmul")]
        { return ldiv16(self, other); }

        #[cfg(feature = "pam")]
        { return pam_div16(self, other); }

        unreachable!("No valid feature selected for div(u16)");
    }
}

impl PlainSub for u16 {
    fn sub(self, other: Self) -> Self {
        #[cfg(feature = "exact")]
        { return canonical_sub16(self, other); }

        #[cfg(any(feature = "lmul", feature = "pam"))]
        { return sub16(self, other); }

        unreachable!("No valid feature selected for sub(u16)");
    }
}

impl PlainTanh for u16 {
    fn tanh(self) -> (Self, Self) {
        #[cfg(feature = "exact")]
        { return canonical_tanh16(self); }

        #[cfg(feature = "lmul")]
        { return lmul_tanh16(self); }

        #[cfg(feature = "pam")]
        { return pam_tanh16(self); }

        unreachable!("No valid feature selected for tanh(u16)");
    }
}

impl PlainReLU for u16 {
    fn relu(self) -> (Self, Self) {
        relu16(self) // same for all features
    }
}

impl PlainBackwardReLU for u16 {
    fn backward_relu(self, grad_output: Self) -> Self {
        backward_relu16(self, grad_output) // same for all features
    }
}

impl PlainSqrt for u16 {
    fn sqrt(self) -> Self {
        sqrt16(self) // same for all features
    }
}

