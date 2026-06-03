mod blood_mnist;
mod breast_cancer;
mod cifar10;
mod dataset_helper;
mod experiment_1_2;
mod experiment_3;
mod f_mnist_exp;
mod he_securenet;
mod mnist_exp;
mod ops_playground;
mod plain_nn_builder;
mod skin_cancer_mnist;
mod tfhe_nn_builder;

#[allow(unused_imports)]
use crate::blood_mnist::blood_mnist_settings::*;
#[allow(unused_imports)]
use crate::breast_cancer::breast_cancer_settings::*;
#[allow(unused_imports)]
use crate::cifar10::cifar10_exp_settings;
#[allow(unused_imports)]
use crate::cifar10::cifar10_exp_settings::*;
#[allow(unused_imports)]
use crate::experiment_1_2::exp_1_settings::*;
#[allow(unused_imports)]
use crate::experiment_1_2::exp_2_settings::*;
#[allow(unused_imports)]
use crate::experiment_3::exp_3_settings::*;
#[allow(unused_imports)]
use crate::f_mnist_exp::f_mnist_exp_settings::*;
#[allow(unused_imports)]
use crate::he_securenet::he_securenet_settings::*;
#[allow(unused_imports)]
use crate::mnist_exp::mnist_exp_settings::*;
#[allow(unused_imports)]
use crate::ops_playground::test_ops::*;
#[allow(unused_imports)]
use crate::skin_cancer_mnist::sc_mnist_settings::*;
use clap::{Parser, ValueEnum};
use dataset_helper::ensure_dataset_exists;
use dialoguer::{Input, Select, theme::ColorfulTheme};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long)]
    experiment: Option<Experiment>,

    #[arg(short, long, value_enum, default_value_t = RunMode::Both)]
    mode: RunMode,

    #[arg(short, long, default_value_t = false)]
    fast_run: bool,

    // --- Playground Parameters ---
    #[arg(long, value_enum)]
    op: Option<OpType>,

    #[arg(long)]
    chain: Option<u32>,

    #[arg(long, value_enum)]
    device: Option<Device>,

    /// Floating-point format precision/configuration
    #[arg(long, value_enum)]
    fp_format: Option<FpFormat>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum FpFormat {
    Fp8,
    Fp16,
    Fp32,
    Fp64,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum OpType {
    Add,
    SsAdd,
    Sub,
    Lmul,
    Ldiv,
    LmulTanh,
    PamMul,
    PamDiv,
    PamTanh,
    Relu,
    Sqrt,
    Log2,
    Misc,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Device {
    Cpu,
    Gpu,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Experiment {
    MlpTernaryMnist,
    CnnTernaryMnist,
    MlpFashionMnist,
    LeNet5FashionMnist,
    CnnDermaMnist,
    CnnBloodMnist,
    ResNet20Cifar10,
    MlpBreastCancer,
    HeSecurenetMlpMnist,
    HeSecurenetCnnMnist,
    EncryptedOps,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum RunMode {
    Plain,
    Both,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // We capture if the user is in interactive mode so we know whether to prompt for sub-menus
    let is_interactive = cli.experiment.is_none();

    // 1. Top-Level Menu Selection
    let selected_experiment = match cli.experiment {
        Some(exp) => exp,
        None => {
            let options = vec![
                "1. MLP[87] on Ternary-MNIST (Plain or Encrypted)",
                "2. CNN[109] on Ternary-MNIST (Plain or Encrypted)",
                "3. MLP[159k] on Fashion-MNIST (Plain)",
                "4. LeNet-5[62k] on Fashion-MNIST (Plain)",
                "5. CNN[366k] on Derma-MNIST (Plain)",
                "6. CNN[366k] on Blood-MNIST (Plain)",
                "7. ResNet-20[273k] on CIFAR-10 (Plain)",
                "8. MLP[650] on Breast Cancer (Plain)",
                "9. HE-Securenet MLP[118k] on MNIST (Plain)",
                "10. HE-Securenet CNN[127k] on MNIST (Plain)",
                "11. Encrypted Ops Playground (Test Encrypted Operations with Custom Configurations)",
            ];

            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Use arrow keys to select an experiment to reproduce")
                .default(0)
                .items(&options)
                .interact()?;

            match selection {
                0 => Experiment::MlpTernaryMnist,
                1 => Experiment::CnnTernaryMnist,
                2 => Experiment::MlpFashionMnist,
                3 => Experiment::LeNet5FashionMnist,
                4 => Experiment::CnnDermaMnist,
                5 => Experiment::CnnBloodMnist,
                6 => Experiment::ResNet20Cifar10,
                7 => Experiment::MlpBreastCancer,
                8 => Experiment::HeSecurenetMlpMnist,
                9 => Experiment::HeSecurenetCnnMnist,
                10 => Experiment::EncryptedOps,
                _ => unreachable!(),
            }
        }
    };

    // 2. Resolve the Execution Mode for Experiments 1 & 2
    let run_mode = if (selected_experiment == Experiment::MlpTernaryMnist
        || selected_experiment == Experiment::CnnTernaryMnist)
        && is_interactive
    {
        let sub_options = vec![
            "Run Plain version only",
            "Run Encrypted version only",
            "Run Both versions sequentially",
        ];

        let sub_selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Choose execution variant")
            .default(2) // Default to 'Both'
            .items(&sub_options)
            .interact()?;

        match sub_selection {
            0 => RunMode::Plain,
            1 => RunMode::Both,
            _ => unreachable!(),
        }
    } else {
        // Fall back to CLI argument flag if passed directly via terminal script
        cli.mode
    };

    // 3. Execution Pipeline
    println!("\n========================================");

    match selected_experiment {
        Experiment::MlpTernaryMnist => {
            ensure_dataset_exists(
                "res/experiment_1_2",
                "https://zenodo.org/records/20510846/files/ternary_mnist.zip?download=1",
            )?;

            println!("Starting MLP Ternary-MNIST Experiment...");
            match run_mode {
                RunMode::Plain => {
                    println!("-> Running Plain Variant");
                    experiment_1_fp32(true, false, true)?;
                }
                RunMode::Both => {
                    println!("-> Running Encrypted Variant (TFHE)");
                    experiment_1_fp32(true, true, true)?;
                }
            }
        }
        Experiment::CnnTernaryMnist => {
            ensure_dataset_exists(
                "res/experiment_1_2",
                "https://zenodo.org/records/20510846/files/ternary_mnist.zip?download=1",
            )?;

            println!("Starting CNN Ternary-MNIST Experiment...");
            match run_mode {
                RunMode::Plain => {
                    println!("-> Running Plain Variant");
                    experiment_2_fp32(true, false, true)?;
                }
                RunMode::Both => {
                    println!("-> Running both variants");
                    experiment_2_fp32(true, true, true)?;
                }
            }
        }
        Experiment::MlpFashionMnist => {
            ensure_dataset_exists(
                "res/f_mnist",
                "https://zenodo.org/records/20510846/files/fashion_mnist.zip?download=1",
            )?;

            println!("Starting MLP Fashion-MNIST Experiment...");
            fmnist_exp1_fp32()?;
        }
        Experiment::LeNet5FashionMnist => {
            ensure_dataset_exists(
                "res/f_mnist",
                "https://zenodo.org/records/20510846/files/fashion_mnist.zip?download=1",
            )?;

            println!("Starting LeNet-5 Fashion-MNIST Experiment...");
            fmnist_exp2_fp32()?;
        }
        Experiment::CnnDermaMnist => {
            ensure_dataset_exists(
                "res/skin_cancer_mnist",
                "https://zenodo.org/records/20510846/files/skin_cancer_mnist.zip?download=1",
            )?;

            println!("Starting CNN Derma-MNIST Experiment...");
            sc_mnist_exp_fp32()?;
        }
        Experiment::CnnBloodMnist => {
            ensure_dataset_exists(
                "res/blood_mnist",
                "https://zenodo.org/records/20510846/files/blood_mnist.zip?download=1",
            )?;

            println!("Starting CNN Blood-MNIST Experiment...");
            blood_vgg_fp32()?;
        }
        Experiment::ResNet20Cifar10 => {
            ensure_dataset_exists(
                "res/cifar10",
                "https://zenodo.org/records/20510846/files/cifar10.zip?download=1",
            )?;

            println!("Starting ResNet-20 CIFAR-10 Experiment...");
            resnet20_fp32()?;
        }
        Experiment::MlpBreastCancer => {
            ensure_dataset_exists(
                "res/breast_cancer",
                "https://zenodo.org/records/20510846/files/breast_cancer.zip?download=1",
            )?;

            println!("Starting MLP Breast Cancer Experiment...");
            breast_cancer_fp32()?;
        }
        Experiment::HeSecurenetMlpMnist => {
            ensure_dataset_exists(
                "res/mnist",
                "https://zenodo.org/records/20510846/files/mnist.zip?download=1",
            )?;

            println!("Starting HE-Securenet MLP MNIST Experiment...");
            securenet_exp1_fp32()?;
        }
        Experiment::HeSecurenetCnnMnist => {
            ensure_dataset_exists(
                "res/mnist",
                "https://zenodo.org/records/20510846/files/mnist.zip?download=1",
            )?;

            println!("Starting HE-Securenet CNN MNIST Experiment...");
            securenet_exp2_fp32()?;
        }
        Experiment::EncryptedOps => {
            println!("Initializing Encrypted Operations Playground...");

            // 1. Resolve Operation Type
            let selected_op = match cli.op {
                Some(op) => op,
                None if is_interactive => {
                    let ops = vec![
                        "Addition",
                        "Same-Sign Addition",
                        "Subtraction",
                        "Lmul",
                        "Ldiv",
                        "Lmul Tanh",
                        "PAM Multiplication",
                        "PAM Division",
                        "PAM Tanh",
                        "ReLU Activation",
                        "Square Root",
                        "Base-2 Logarithm",
                        "Miscellaneous",
                    ];
                    let selection = Select::with_theme(&ColorfulTheme::default())
                        .with_prompt("Select the encrypted operation to test")
                        .default(0)
                        .items(&ops)
                        .interact()?;
                    match selection {
                        0 => OpType::Add,
                        1 => OpType::SsAdd,
                        2 => OpType::Sub,
                        3 => OpType::Lmul,
                        4 => OpType::Ldiv,
                        5 => OpType::LmulTanh,
                        6 => OpType::PamMul,
                        7 => OpType::PamDiv,
                        8 => OpType::PamTanh,
                        9 => OpType::Relu,
                        10 => OpType::Sqrt,
                        11 => OpType::Log2,
                        12 => OpType::Misc,
                        _ => unreachable!(),
                    }
                }
                None => OpType::Lmul,
            };

            let precision_bits: u32 = match selected_op {
                OpType::Misc => {
                    println!("-> 'Miscellaneous' selected.");
                    32
                }
                // For all other operations, execute your normal selection logic
                _ => match cli.fp_format {
                    Some(FpFormat::Fp8) => 8,
                    Some(FpFormat::Fp16) => 16,
                    Some(FpFormat::Fp32) => 32,
                    Some(FpFormat::Fp64) => 64,
                    None if is_interactive => {
                        let formats = vec![
                            "FP8 (8-bit)",
                            "FP16 (16-bit)",
                            "FP32 (32-bit)",
                            "FP64 (64-bit)",
                        ];
                        let selection = Select::with_theme(&ColorfulTheme::default())
                            .with_prompt("Select Floating-Point Format")
                            .default(2) // Updated to index 2 so FP32 is highlighted by default
                            .items(&formats)
                            .interact()?;
                        match selection {
                            0 => 8,
                            1 => 16,
                            2 => 32,
                            3 => 64,
                            _ => unreachable!(),
                        }
                    }
                    None => 32,
                },
            };

            // 3. Resolve Chain Length
            let chain_length: u32 = match cli.chain {
                Some(c) => c,
                None if is_interactive => Input::with_theme(&ColorfulTheme::default())
                    .with_prompt("Enter the number of chained operations")
                    .default(10)
                    .interact_text()?,
                None => 10,
            };

            // 4. Resolve Hardware Device
            let selected_device = match cli.device {
                Some(d) => d,
                None if is_interactive => {
                    let devices = vec!["CPU", "GPU (CUDA Backend)"];
                    let selection = Select::with_theme(&ColorfulTheme::default())
                        .with_prompt("Select hardware backend")
                        .default(1)
                        .items(&devices)
                        .interact()?;
                    match selection {
                        0 => Device::Cpu,
                        1 => Device::Gpu,
                        _ => unreachable!(),
                    }
                }
                None => Device::Gpu,
            };

            // Map variables for your processing function
            let op_str = match selected_op {
                OpType::Add => "add",
                OpType::SsAdd => "same_sign_add",
                OpType::Sub => "sub",
                OpType::Lmul => "lmul",
                OpType::Ldiv => "ldiv",
                OpType::LmulTanh => "lmul_tanh",
                OpType::PamMul => "pam_mul",
                OpType::PamDiv => "pam_div",
                OpType::PamTanh => "pam_tanh",
                OpType::Relu => "relu",
                OpType::Sqrt => "sqrt",
                OpType::Log2 => "log2",
                OpType::Misc => "misc",
            };
            let is_gpu = selected_device == Device::Gpu;

            println!(
                "\n-> Executing {} chained `{}` operations using {} format on {}...",
                chain_length,
                op_str,
                format!("FP{}", precision_bits),
                if is_gpu { "GPU" } else { "CPU" }
            );

            test_encrypted_ops(
                op_str,
                precision_bits as usize,
                is_gpu,
                chain_length as usize,
                -10.0,
                10.0,
            )?;
        }
    }

    println!("========================================\nDone.");
    Ok(())
}
