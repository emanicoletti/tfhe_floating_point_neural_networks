mod tfhe_nn_builder;
mod plain_nn_builder;
mod experiment_1_2;
mod experiment_3;
mod mnist_exp;
mod f_mnist_exp;
mod ops_playground;

#[allow(unused_imports)]
use crate::experiment_1_2::exp_1_settings::*;
#[allow(unused_imports)]
use crate::experiment_1_2::exp_2_settings::*;
#[allow(unused_imports)]
use crate::experiment_3::exp_3_settings::*;
use crate::f_mnist_exp::f_mnist_exp_settings::*;
#[allow(unused_imports)]
use crate::ops_playground::test_ops::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    //test_encrypted_ops("test", 64, true, 10, -5.0, 5.0)?;
    //experiment_1_fp32(true, false, true)?;
    fmnist_exp1_fp32(true, false, false)?;
    //experiment_3_fp32()?;

    Ok(())
}






