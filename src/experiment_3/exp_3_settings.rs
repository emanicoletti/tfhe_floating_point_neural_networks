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

    Ok(())
}