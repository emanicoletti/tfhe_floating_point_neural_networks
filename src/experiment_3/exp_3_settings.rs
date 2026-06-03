use crate::plain_nn_builder::plain_nn::{
    PlainNeuralNetwork, PlainNeuralNetworkU16, PlainNeuralNetworkU32,
};
use crate::plain_nn_builder::plain_utils::load_from_file::*;

#[allow(dead_code)]
pub fn experiment_3_fp32() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from folder experiment_3
    let (train_inputs_arr, train_labels_arr, _, _, val_inputs_arr, val_labels_arr) = load_data(3)?;

    // Format the input accordingly
    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let val_inputs = array2_to_vecvec(&val_inputs_arr);
    let val_labels = array2_to_vecvec(&val_labels_arr);

    // Create the plain model
    let mut plain_model = PlainNeuralNetworkU32::create(Some(3));

    // Define the architecture
    plain_model.add_conv(1, 32, 3, 3, 1, 1);
    plain_model.add_batch_norm(32);
    plain_model.add_relu_activation(32);
    plain_model.add_max_pooling(vec![28, 28], 2, 2);

    plain_model.add_conv(32, 64, 3, 3, 1, 1);
    plain_model.add_batch_norm(64);
    plain_model.add_relu_activation(64);
    plain_model.add_max_pooling(vec![14, 14], 2, 2);

    plain_model.add_conv(64, 128, 3, 3, 1, 1);
    plain_model.add_batch_norm(128);
    plain_model.add_relu_activation(128);

    plain_model.add_dense(6272, 256);
    plain_model.add_relu_activation(256);
    plain_model.add_dense(256, 10);

    // Train and validate the plain model
    plain_model.train_and_validate(
        50,
        64,
        0.01,
        0.0,
        0.0,
        &train_inputs,
        &train_labels,
        &val_inputs,
        &val_labels,
        vec![6400, 1, 28, 28],
        vec![6400, 1, 1, 10],
        vec![1000, 1, 28, 28],
        vec![1000, 1, 1, 10],
    );

    Ok(())
}

#[allow(dead_code)]
pub fn experiment_3_fp16() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from folder experiment_3
    let (train_inputs_arr, train_labels_arr, _, _, val_inputs_arr, val_labels_arr) = load_data(3)?;

    // Format the input accordingly
    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let val_inputs = array2_to_vecvec(&val_inputs_arr);
    let val_labels = array2_to_vecvec(&val_labels_arr);

    // Create the plain model
    let mut plain_model = PlainNeuralNetworkU16::create(Some(3));

    // Define the architecture
    plain_model.add_conv(1, 32, 3, 3, 1, 1);
    plain_model.add_batch_norm(32);
    plain_model.add_relu_activation(32);
    plain_model.add_max_pooling(vec![28, 28], 2, 2);

    plain_model.add_conv(32, 64, 3, 3, 1, 1);
    plain_model.add_batch_norm(64);
    plain_model.add_relu_activation(64);
    plain_model.add_max_pooling(vec![14, 14], 2, 2);

    plain_model.add_conv(64, 128, 3, 3, 1, 1);
    plain_model.add_batch_norm(128);
    plain_model.add_relu_activation(128);

    plain_model.add_dense(6272, 256);
    plain_model.add_relu_activation(256);
    plain_model.add_dense(256, 10);

    // Train and validate the plain model
    plain_model.train_and_validate(
        25,
        64,
        0.1,
        0.0,
        0.0,
        &train_inputs,
        &train_labels,
        &val_inputs,
        &val_labels,
        vec![6400, 1, 28, 28],
        vec![6400, 1, 1, 10],
        vec![1000, 1, 28, 28],
        vec![1000, 1, 1, 10],
    );

    Ok(())
}
