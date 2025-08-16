use rand_distr::num_traits::float;
use tfhe::boolean::backward_compatibility::server_key;
use tfhe::shortint::parameters::*;
use tfhe::shortint::parameters::v1_2::*;
use tfhe::prelude::*;
use tfhe::shortint::client_key;
use tfhe::{set_server_key, generate_keys, ConfigBuilder, FheUint8, FheUint16, FheUint32, FheUint64, ClientKey, ServerKey, CompressedServerKey, CudaServerKey};
use std::time::Instant;
use rand::Rng;
use half::f16;
use tfhe::prelude::*;

use ndarray::Array2;
use ndarray_npy::read_npy;
use std::path::Path;
use std::error::Error;

use crate::plain_nn_builder::plain_ops::{add16, lmul16, same_sign_add16, log2_u16, ldiv16, log2_u32, sqrt_u16, sqrt_u32};
use crate::tfhe_nn_builder::add::fhe_add16_gpu;
use crate::tfhe_nn_builder::mul::fhe_lmul16_gpu;
use crate::tfhe_nn_builder::same_sign_add::fhe_ss_add16_gpu;
use crate::tfhe_nn_builder::sqrt::*;
use crate::tfhe_nn_builder::encrypted_nn::{EncryptedNeuralNetwork, EncryptedNeuralNetworkU32GPU, EncryptedNeuralNetworkU16GPU};
use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32, PlainNeuralNetworkU16};

mod tfhe_nn_builder;
mod plain_nn_builder;
mod experiment_1_2;


fn main() -> Result<(), Box<dyn std::error::Error>> {

    let config =
        ConfigBuilder::with_custom_parameters(PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
    let client_key = ClientKey::generate(config);
    let compressed_server_key = CompressedServerKey::new(&client_key);
    let gpu_key = compressed_server_key.decompress_to_gpu();

    let mut total_duration = std::time::Duration::new(0, 0);

    let mut rng = rand::thread_rng();
    for _ in 0..1{

        //let float_a: f32 = rng.gen_range(0.0..5.0);
        let float_a: f32 = 0.00000005; // Fixed value for testing

        let float_a_f16 = f16::from_f32(float_a);

        // Convert f16 to u16 (bit pattern)
        let clear_a: u16 = float_a_f16.to_bits();

        // Encrypting the input data using the (private) client_key
        let encrypted_a = FheUint16::try_encrypt(clear_a, &client_key)?; 
        let encrypted_zero = FheUint16::try_encrypt(0u16, &client_key)?;

        let start = Instant::now();

        //let encrypted_multiply = fhe_lmul16_parallel(encrypted_a, encrypted_b, encrypted_zero.clone(), gpu_key.clone());
        let encrypted_accumulate = fhe_sqrt16_gpu(encrypted_a.clone(), encrypted_zero.clone(), gpu_key.clone());

        // Add the execution time to the total
        total_duration += start.elapsed();

        let clear_res: u16 = encrypted_accumulate.decrypt(&client_key);

        let float_res = f16::from_bits(clear_res).to_f32();

        println!("sqrt({:?}) = {:?}, real: {:?}", float_a, float_res, f16::from_bits(sqrt_u16(clear_a)));
        println!("Execution time: {:?}", total_duration);

    }

    /* 
    let (train_inputs_arr, train_labels_arr, val_inputs_arr, val_labels_arr, test_inputs_arr, test_labels_arr) = load_data()?;

    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let val_inputs = array2_to_vecvec(&val_inputs_arr);
    let val_labels = array2_to_vecvec(&val_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    let mut plain_model = PlainNeuralNetworkU32::create();
    plain_model.add_max_pooling(vec![16, 16], 4, 4, 0);
    plain_model.add_conv(1, 1, 2, 2);
    plain_model.add_relu_activation(4);
    plain_model.add_dense(4, 3);

    plain_model.print_plain_weights(String::from("Conv2"));
    plain_model.print_plain_biases(String::from("Conv2"));
    plain_model.print_plain_weights(String::from("Dense4"));
    plain_model.print_plain_biases(String::from("Dense4"));

    plain_model.train(
        1,
        5,
        0.1,
        &train_inputs,
        &train_labels,
        vec![5, 1, 16, 16],
        vec![5, 1, 1, 3],
    );

    plain_model.print_plain_weights(String::from("Conv2"));
    plain_model.print_plain_biases(String::from("Conv2"));
    plain_model.print_plain_weights(String::from("Dense4"));
    plain_model.print_plain_biases(String::from("Dense4"));

    

    let mut model = EncryptedNeuralNetworkU32GPU::create();
    model.add_max_pooling(vec![16, 16], 4, 4, 0);
    model.add_conv(1, 1, 2, 2);
    model.add_relu_activation(4);
    model.add_dense(4, 3);

    model.print_plain_weights(String::from("Conv2"));
    model.print_plain_biases(String::from("Conv2"));
    model.print_plain_weights(String::from("Dense4"));
    model.print_plain_biases(String::from("Dense4"));

    model.train(
        1,
        5,
        0.1,
        &train_inputs,
        &train_labels,
        vec![5, 1, 16, 16],
        vec![5, 1, 1, 3],
    );

    model.print_plain_weights(String::from("Conv2"));
    model.print_plain_biases(String::from("Conv2"));
    model.print_plain_weights(String::from("Dense4"));
    model.print_plain_biases(String::from("Dense4"));
    
    let mut plain_model = PlainNeuralNetworkU16::create();
    plain_model.add_max_pooling(vec![16, 16], 4, 4, 0);
    plain_model.add_dense(16, 4);
    plain_model.add_tanh_activation(4);
    plain_model.add_dense(4, 2);
    plain_model.add_tanh_activation(2);
    plain_model.add_dense(2, 3);
    //plain_model.add_tanh_activation(3);

    let id = String::from("Dense2");
    let id1 = String::from("Dense4");
    let id2 = String::from("Dense6");
    plain_model.print_plain_weights(id.clone());
    plain_model.print_plain_biases(id.clone());
    plain_model.print_plain_weights(id1.clone());
    plain_model.print_plain_biases(id1.clone());
    plain_model.print_plain_weights(id2.clone());
    plain_model.print_plain_biases(id2.clone());

    plain_model.train(
        1,
        5,
        0.1,
        &train_inputs,
        &train_labels,
        vec![5, 1, 16, 16],
        vec![5, 1, 1, 3],
    );

    plain_model.print_plain_weights(id.clone());
    plain_model.print_plain_biases(id.clone());
    plain_model.print_plain_weights(id1.clone());
    plain_model.print_plain_biases(id1.clone());
    plain_model.print_plain_weights(id2.clone());
    plain_model.print_plain_biases(id2.clone());
    
    
    let mut model = EncryptedNeuralNetworkU16GPU::create();
    model.add_max_pooling(vec![16, 16], 4, 4, 0);
    model.add_dense(16, 4);
    model.add_tanh_activation(4);
    model.add_dense(4, 2);
    model.add_tanh_activation(2);
    model.add_dense(2, 3);
    //model.add_tanh_activation(3);
    
    //model.add_dense(10, 3);
    let id = String::from("Dense2");
    let id1 = String::from("Dense4");
    let id2 = String::from("Dense6");
    model.print_plain_weights(id.clone());
    model.print_plain_biases(id.clone());
    model.print_plain_weights(id1.clone());
    model.print_plain_biases(id1.clone());
    model.print_plain_weights(id2.clone());
    model.print_plain_biases(id2.clone());
    
    
    model.train(
        1,                          
        5,                           
        0.1,                       
        &train_inputs,              
        &train_labels,             
        vec![5, 1, 16, 16],
        vec![5, 1, 1, 3]          
    );
    


    model.print_plain_weights(id.clone());
    model.print_plain_biases(id.clone());
    model.print_plain_weights(id1.clone());
    model.print_plain_biases(id1.clone());
    model.print_plain_weights(id2.clone());
    model.print_plain_biases(id2.clone());
    
    let mut correct = 0;
    let total = val_labels.len();

    for (input, label) in val_inputs.iter().zip(val_labels.iter()) {
        let input_batch = vec![input.clone()]; // batch size = 1
        let input_shape = vec![1, 1, 16, 16];
        let label_shape = vec![1, 1, 1, 3];

        let prediction = plain_model.inference(&input_batch, input_shape.clone(), label_shape.clone());
        //let prediction_enc = model.inference(&input_batch, input_shape.clone(), label_shape.clone());

        //println!("Prediction: {:?}", prediction);
        //println!("Encrypted Prediction: {:?}", prediction_enc);

        // Get predicted class (argmax)
        let predicted_class = prediction
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // Get actual class from one-hot label
        let actual_class = label
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if predicted_class == actual_class {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / total as f32;
    println!("Validation Accuracy: {:.2}%", accuracy * 100.0);

    let mut correct = 0;
    let total = test_labels.len();

    for (input, label) in test_inputs.iter().zip(test_labels.iter()) {
        let input_batch = vec![input.clone()]; // batch size = 1
        let input_shape = vec![1, 1, 16, 16];
        let label_shape = vec![1, 1, 1, 3];

        let prediction = plain_model.inference(&input_batch, input_shape, label_shape);

        // Get predicted class (argmax)
        let predicted_class = prediction
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // Get actual class from one-hot label
        let actual_class = label
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if predicted_class == actual_class {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / total as f32;
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);
    */

    Ok(())
}

/// Loads training data from .npy files
fn load_data() -> Result<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>), Box<dyn Error>> {
    let x_train: Array2<f32> = read_npy(Path::new("src/experiment_1_2/dataset/x_train.npy"))?;
    let y_train: Array2<f32> = read_npy(Path::new("src/experiment_1_2/dataset/y_train.npy"))?;
    let x_val: Array2<f32> = read_npy(Path::new("src/experiment_1_2/dataset/x_val.npy"))?;
    let y_val: Array2<f32> = read_npy(Path::new("src/experiment_1_2/dataset/y_val.npy"))?;
    let x_test: Array2<f32> = read_npy(Path::new("src/experiment_1_2/dataset/x_test.npy"))?;
    let y_test: Array2<f32> = read_npy(Path::new("src/experiment_1_2/dataset/y_test.npy"))?;
    Ok((x_train, y_train, x_val, y_val, x_test, y_test))
}

fn array2_to_vecvec(array: &Array2<f32>) -> Vec<Vec<f32>> {
    array
        .outer_iter() // iterates over rows
        .map(|row| row.to_vec())
        .collect()
}