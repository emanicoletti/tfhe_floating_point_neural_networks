use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32};
use crate::plain_nn_builder::plain_utils::load_from_file::*;

#[allow(dead_code)]
pub fn securenet_exp1_fp32() -> Result<(), Box<dyn std::error::Error>> {

    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) = load_data(5)?;

    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    // Declare the plain model structure
    let mut plain_model = PlainNeuralNetworkU32::create(Some(11));

    // Define the architecture
    plain_model.add_dense(784, 128);
    plain_model.add_relu_activation(128);
    plain_model.add_dense(128, 128);
    plain_model.add_relu_activation(128);
    plain_model.add_dense(128, 10);

    plain_model.train_and_validate(
        25,
        128,
        0.5,
        0.0,
        0.0,
        &train_inputs,
        &train_labels,
        &test_inputs,
        &test_labels,
        vec![50000, 1, 28, 28],
        vec![50000, 1, 1, 10],
        vec![10000, 1, 28, 28],
        vec![10000, 1, 1, 10],
        );

    Ok(())
}

pub fn securenet_exp2_fp32() -> Result<(), Box<dyn std::error::Error>> {

    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) = load_data(5)?;

    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    // Declare the plain model structure
    let mut plain_model = PlainNeuralNetworkU32::create(Some(12));

    // Define the architecture// Esempio con input_shape = (altezza, larghezza, 3)
    plain_model.add_conv(1, 5, 5, 5, 2, 2);
    plain_model.add_relu_activation(980);
    plain_model.add_dense(980, 128);
    plain_model.add_relu_activation(128);
    plain_model.add_dense(128, 10);

    plain_model.train_and_validate(
        25,
        128,
        0.5,
        0.0,
        0.0,
        &train_inputs,
        &train_labels,
        &test_inputs,
        &test_labels,
        vec![50000, 1, 28, 28],
        vec![50000, 1, 1, 10],
        vec![10000, 1, 28, 28],
        vec![10000, 1, 1, 10],
        );

    Ok(())
}