use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32};
use crate::plain_nn_builder::plain_utils::load_from_file::*;

#[allow(dead_code)]
pub fn fmnist_exp1_fp32() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from folder
    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) =
        load_data(5)?;

    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    // Declare the plain model structure
    let mut plain_model = PlainNeuralNetworkU32::create(Some(5));

    // Define the architecture
    plain_model.add_dense(784, 200);
    plain_model.add_tanh_activation(200);
    plain_model.add_dense(200, 10);
    plain_model.add_tanh_activation(10);

    plain_model.train_and_validate(
        20,
        50,
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

#[allow(dead_code)]
pub fn fmnist_exp2_fp32() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from folder
    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) =
        load_data(5)?;

    // Format the input accordingly
    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    // Declare the plain model structure
    let mut plain_model = PlainNeuralNetworkU32::create(Some(6));

    // Define the architecture
    plain_model.add_conv(1, 6, 5, 5, 1, 2);
    plain_model.add_relu_activation(4704);
    plain_model.add_max_pooling(vec![1, 6, 28, 28], 2, 2);
    plain_model.add_conv(6, 16, 5, 5, 1, 0);
    plain_model.add_relu_activation(1600);
    plain_model.add_max_pooling(vec![1, 16, 10, 10], 2, 2);
    plain_model.add_dense(16 * 5 * 5, 120);
    plain_model.add_relu_activation(120);
    plain_model.add_dense(120, 84);
    plain_model.add_relu_activation(84);
    plain_model.add_dense(84, 10);

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
