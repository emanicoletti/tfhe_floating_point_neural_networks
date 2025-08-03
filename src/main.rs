use tfhe::boolean::backward_compatibility::server_key;
use tfhe::shortint::parameters::*;
use tfhe::shortint::parameters::v1_2::*;
use tfhe::{set_server_key};
use tfhe::{ConfigBuilder, ClientKey, generate_keys, CompressedServerKey, CudaServerKey, FheUint8, FheUint16, FheUint32, FheUint64};
use std::time::Instant;
use rand::Rng;
use half::f16;
use tfhe::prelude::*;

use ndarray::Array2;
use ndarray_npy::read_npy;
use std::path::Path;
use std::error::Error;

use crate::tfhe_nn_builder::encrypted_nn::{EncryptedNeuralNetwork, EncryptedNeuralNetworkU32GPU};
use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32};

mod tfhe_nn_builder;
mod plain_nn_builder;

use rayon::ThreadPoolBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let (train_inputs_arr, train_labels_arr, val_inputs_arr, val_labels_arr, test_inputs_arr, test_labels_arr) = load_data()?;

    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let val_inputs = array2_to_vecvec(&val_inputs_arr);
    let val_labels = array2_to_vecvec(&val_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    
    let mut model = EncryptedNeuralNetworkU32GPU::create();
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
        1,                           
        0.1,                       
        &train_inputs,              
        &train_labels,             
        vec![1, 1, 16, 16],
        vec![1, 1, 1, 3]          
    );
    


    model.print_plain_weights(id.clone());
    model.print_plain_biases(id.clone());
    model.print_plain_weights(id1.clone());
    model.print_plain_biases(id1.clone());
    model.print_plain_weights(id2.clone());
    model.print_plain_biases(id2.clone());
    
    
    let mut plain_model = PlainNeuralNetworkU32::create();
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
        1,                           
        0.1,                       
        &train_inputs,              
        &train_labels,             
        vec![1, 1, 16, 16],
        vec![1, 1, 1, 3]          
    );

    plain_model.print_plain_weights(id.clone());
    plain_model.print_plain_biases(id.clone());
    plain_model.print_plain_weights(id1.clone());
    plain_model.print_plain_biases(id1.clone());
    plain_model.print_plain_weights(id2.clone());
    plain_model.print_plain_biases(id2.clone());


    /* 
    let mut correct = 0;
    let total = val_labels.len();

    for (input, label) in val_inputs.iter().zip(val_labels.iter()) {
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
    let x_train: Array2<f32> = read_npy(Path::new("src/Experiment_1/dataset/x_train.npy"))?;
    let y_train: Array2<f32> = read_npy(Path::new("src/Experiment_1/dataset/y_train.npy"))?;
    let x_val: Array2<f32> = read_npy(Path::new("src/Experiment_1/dataset/x_val.npy"))?;
    let y_val: Array2<f32> = read_npy(Path::new("src/Experiment_1/dataset/y_val.npy"))?;
    let x_test: Array2<f32> = read_npy(Path::new("src/Experiment_1/dataset/x_test.npy"))?;
    let y_test: Array2<f32> = read_npy(Path::new("src/Experiment_1/dataset/y_test.npy"))?;
    Ok((x_train, y_train, x_val, y_val, x_test, y_test))
}

fn array2_to_vecvec(array: &Array2<f32>) -> Vec<Vec<f32>> {
    array
        .outer_iter() // iterates over rows
        .map(|row| row.to_vec())
        .collect()
}