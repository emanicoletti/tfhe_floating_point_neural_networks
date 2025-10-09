use crate::tfhe_nn_builder::encrypted_nn::{EncryptedNeuralNetwork, EncryptedNeuralNetworkU32GPU, EncryptedNeuralNetworkU16GPU};
use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32, PlainNeuralNetworkU16};
use crate::plain_nn_builder::plain_utils::load_from_file::*;

pub fn experiment_3_fp32(train_plain_network: bool, test_plain_network: bool, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {

    // Validate input parameters
    if !train_plain_network && test_plain_network {
        panic!("Cannot test plain network without training it first");
    }

    /* Load data from folder experiment_1_2 */
    let (train_inputs_arr, train_labels_arr, val_inputs_arr, val_labels_arr, test_inputs_arr, test_labels_arr) = load_data(3)?;

    /* Format the input accordingly */
    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let val_inputs = array2_to_vecvec(&val_inputs_arr);
    let val_labels = array2_to_vecvec(&val_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    let mut plain_model = PlainNeuralNetworkU32::create(Some(3));

    plain_model.add_conv(1, 32, 3, 3, 1, 1);
    plain_model.add_batch_norm(32);
    plain_model.add_relu_activation(32);
    plain_model.add_max_pooling(vec![28, 28], 2, 2, 0);

    plain_model.add_conv(32, 64, 3, 3, 1, 1);
    plain_model.add_batch_norm(64);
    plain_model.add_relu_activation(64);
    plain_model.add_max_pooling(vec![14, 14], 2, 2, 0);

    plain_model.add_conv(64, 128, 3, 3, 1, 1);
    plain_model.add_batch_norm(128);
    plain_model.add_relu_activation(128);

    plain_model.add_dense(6272, 256);
    plain_model.add_relu_activation(256);
    plain_model.add_dense(256, 10);

    plain_model.train_and_validate(
        50,
        64,
        0.01,
        &train_inputs,
        &train_labels,
        &test_inputs,
        &test_labels,
        vec![6400, 1, 28, 28],
        vec![6400, 1, 1, 10],
        vec![1000, 1, 28, 28],
        vec![1000, 1, 1, 10],
    );

    /* 
    let mut correct = 0;
    let mut total = test_labels.len();

    for (input, label) in test_inputs.iter().zip(test_labels.iter()) {
        let input_batch = vec![input.clone()]; // batch size = 1
        let input_shape = vec![1, 1, 28, 28];
        let label_shape = vec![1, 1, 1, 10];

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

pub fn experiment_3_fp16(train_plain_network: bool, test_plain_network: bool, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {

    // Validate input parameters
    if !train_plain_network && test_plain_network {
        panic!("Cannot test plain network without training it first");
    }

    /* Load data from folder experiment_1_2 */
    let (train_inputs_arr, train_labels_arr, val_inputs_arr, val_labels_arr, test_inputs_arr, test_labels_arr) = load_data(3)?;

    /* Format the input accordingly */
    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let val_inputs = array2_to_vecvec(&val_inputs_arr);
    let val_labels = array2_to_vecvec(&val_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    let mut plain_model = PlainNeuralNetworkU16::create(Some(3));

    plain_model.add_conv(1, 32, 3, 3, 1, 1);
    plain_model.add_batch_norm(32);
    plain_model.add_relu_activation(32);
    plain_model.add_max_pooling(vec![28, 28], 2, 2, 0);

    plain_model.add_conv(32, 64, 3, 3, 1, 1);
    plain_model.add_batch_norm(64);
    plain_model.add_relu_activation(64);
    plain_model.add_max_pooling(vec![14, 14], 2, 2, 0);

    plain_model.add_conv(64, 128, 3, 3, 1, 1);
    plain_model.add_batch_norm(128);
    plain_model.add_relu_activation(128);

    plain_model.add_dense(6272, 256);
    plain_model.add_relu_activation(256);
    plain_model.add_dense(256, 10);

    plain_model.train_and_validate(
        25,
        64,
        0.1,
        &train_inputs,
        &train_labels,
        &test_inputs,
        &test_labels,
        vec![6400, 1, 28, 28],
        vec![6400, 1, 1, 10],
        vec![1000, 1, 28, 28],
        vec![1000, 1, 1, 10],
    );

    Ok(())
}