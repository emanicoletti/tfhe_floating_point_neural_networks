
mod tfhe_nn_builder;
mod plain_nn_builder;
mod experiment_1_2;
mod experiment_3;
mod mnist_exp;
mod f_mnist_exp;
mod skin_cancer_mnist;
mod blood_mnist;
mod ops_playground;
mod cifar10;
mod breast_cancer;
mod he_securenet;

#[allow(unused_imports)]
use crate::cifar10::cifar10_exp_settings;
#[allow(unused_imports)]
use crate::experiment_1_2::exp_2_settings::*;
#[allow(unused_imports)]
use crate::experiment_3::exp_3_settings::*;
#[allow(unused_imports)]
use crate::f_mnist_exp::f_mnist_exp_settings::*;
#[allow(unused_imports)]
use crate::mnist_exp::mnist_exp_settings::*;
#[allow(unused_imports)]
use crate::skin_cancer_mnist::sc_mnist_settings::*;
#[allow(unused_imports)]
use crate::blood_mnist::blood_mnist_settings::*;
#[allow(unused_imports)]
use crate::cifar10::cifar10_exp_settings::*;
#[allow(unused_imports)]
use crate::he_securenet::he_securenet_settings::*;
#[allow(unused_imports)]
use crate::ops_playground::test_ops::*;
#[allow(unused_imports)]
use crate::breast_cancer::breast_cancer_settings::*;
use crate::plain_nn_builder::plain_ops::ops::*;


#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Uncomment the desired experiment to run

    // Experiment 1 (MLP and CNN on Ternary-MNIST)
    //let _profiler = dhat::Profiler::new_heap();
    //experiment_1_fp32(false, true, false)?;
    // experiment_2_fp32(true, false, true)?;

    // Experiment 2 (MLP and LeNet-5 on Fashion-MNIST)
    // fmnist_exp1_fp32()?;
    // fmnist_exp2_fp32()?;

    // Experiment 3 (VGG-like on Derma-MNIST and Blood-MNIST)
    // sc_mnist_exp_fp32()?;
    // blood_vgg_fp32()?;

    // Experiment 4 (ResNet-20 on CIFAR-10)
    //resnet20_fp32()?;

    // Experiment 5 (MLP on Breast Cancer Wisconsin)
    // breast_cancer_fp32()?;

    // Test encrypted operations
    //test_encrypted_ops("relu", 32, false, 1, -5.0, 5.0)?;

    //securenet_exp1_fp32()?;

    securenet_exp2_fp32()?;

    //fmnist_exp2_fp32()?;

    Ok(())
}






