mod tfhe_nn_builder;
mod plain_nn_builder;
mod experiment_1_2;
mod experiment_3;
mod MNIST_exp;
mod ops_playground;

use crate::MNIST_exp::MNIST_exp_settings::mnist_exp_fp32;
#[allow(unused_imports)]
use crate::experiment_1_2::exp_1_settings::*;
#[allow(unused_imports)]
use crate::experiment_1_2::exp_2_settings::*;
#[allow(unused_imports)]
use crate::experiment_3::exp_3_settings::*;
#[allow(unused_imports)]
use crate::ops_playground::test_ops::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    //test_encrypted_ops("test", 64, true, 10, -5.0, 5.0)?;
    //experiment_1_fp32(true, false, true)?;
    //mnist_exp_fp32(true, false, true)?;
    //experiment_3_fp32()?;

    Ok(())
}






