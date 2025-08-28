use crate::tfhe_nn_builder::encrypted_nn::{EncryptedNeuralNetwork, EncryptedNeuralNetworkU32GPU, EncryptedNeuralNetworkU16GPU};
use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32, PlainNeuralNetworkU16};
use crate::plain_nn_builder::plain_utils::load_from_file::*;

pub fn experiment_2_fp32(train_plain_network: bool, train_encrypted_network: bool, test_plain_network: bool, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {

    // Validate input parameters
    if !train_plain_network && test_plain_network {
        panic!("Cannot test plain network without training it first");
    }
    if !train_plain_network && !train_encrypted_network {
        panic!("At least one of train_plain_network or train_encrypted_network must be true");
    }

    /* Load data from folder experiment_1_2 */
    let (train_inputs_arr, train_labels_arr, val_inputs_arr, val_labels_arr, test_inputs_arr, test_labels_arr) = load_data(2)?;

    /* Format the input accordingly */
    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let val_inputs = array2_to_vecvec(&val_inputs_arr);
    let val_labels = array2_to_vecvec(&val_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    let mut plain_model = PlainNeuralNetworkU32::create(Some(2));

    if train_plain_network {
        // Definition of the plain network architecture
        plain_model.add_max_pooling(vec![16, 16], 4, 4, 0);
        plain_model.add_conv(1, 1, 2, 2, 2, 0);
        plain_model.add_relu_activation(4);
        plain_model.add_dense(4, 3);

        /*
        if verbose -> Print a summary
         */

        plain_model.train(
            1,
            1,
            0.1,
            &train_inputs,
            &train_labels,
            vec![1, 1, 16, 16],
            vec![1, 1, 1, 3],
        );

        plain_model.print_plain_weights("Conv2".to_string());
        plain_model.print_plain_biases("Conv2".to_string());
        plain_model.print_plain_weights("Dense4".to_string());
        plain_model.print_plain_biases("Dense4".to_string());
    }

    if train_encrypted_network {
        // Definition of the plain network architecture
        println!("WARNING: The encrypted training could require days to be completed");

        let mut model = EncryptedNeuralNetworkU32GPU::create(Some(2));
        model.add_max_pooling(vec![16, 16], 4, 4, 0);
        model.add_conv(1, 1, 2, 2, 2, 0);
        model.add_relu_activation(4);
        model.add_dense(4, 3);

        model.train(
            1,
            1,
            0.1,
            &train_inputs,
            &train_labels,
            vec![1, 1, 16, 16],
            vec![1, 1, 1, 3],
        );

        model.print_plain_weights("Conv2".to_string());
        model.print_plain_biases("Conv2".to_string());
        model.print_plain_weights("Dense4".to_string());
        model.print_plain_biases("Dense4".to_string());
    }

    if test_plain_network {
            
        let mut correct = 0;
        let mut total = val_labels.len();

        for (input, label) in val_inputs.iter().zip(val_labels.iter()) {
            let input_batch = vec![input.clone()]; // batch size = 1
            let input_shape = vec![1, 1, 16, 16];
            let label_shape = vec![1, 1, 1, 3];

            let prediction = plain_model.inference(&input_batch, input_shape.clone(), label_shape.clone());

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

        correct = 0;
        total = test_labels.len();

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

    }

    Ok(())

}

pub fn experiment_2_fp16(train_plain_network: bool, train_encrypted_network: bool, test_plain_network: bool, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {

    // Validate input parameters
    if !train_plain_network && test_plain_network {
        panic!("Cannot test plain network without training it first");
    }
    if !train_plain_network && !train_encrypted_network {
        panic!("At least one of train_plain_network or train_encrypted_network must be true");
    }

    /* Load data from folder experiment_1_2 */
    let (train_inputs_arr, train_labels_arr, val_inputs_arr, val_labels_arr, test_inputs_arr, test_labels_arr) = load_data(2)?;

    /* Format the input accordingly */
    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let val_inputs = array2_to_vecvec(&val_inputs_arr);
    let val_labels = array2_to_vecvec(&val_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    let mut plain_model = PlainNeuralNetworkU16::create(Some(2));

    if train_plain_network {
        // Definition of the plain network architecture
        plain_model.add_max_pooling(vec![16, 16], 4, 4, 0);
        plain_model.add_conv(1, 1, 2, 2, 2, 0);
        plain_model.add_relu_activation(4);
        plain_model.add_dense(4, 3);

        /*
        if verbose -> Print a summary
         */

        plain_model.print_plain_weights("Conv2".to_string());
        plain_model.print_plain_biases("Conv2".to_string());
        plain_model.print_plain_weights("Dense4".to_string());
        plain_model.print_plain_biases("Dense4".to_string());

        plain_model.train(
            1,
            1,
            0.1,
            &train_inputs,
            &train_labels,
            vec![1, 1, 16, 16],
            vec![1, 1, 1, 3],
        );

        plain_model.print_plain_weights("Conv2".to_string());
        plain_model.print_plain_biases("Conv2".to_string());
        plain_model.print_plain_weights("Dense4".to_string());
        plain_model.print_plain_biases("Dense4".to_string());
    }

    if train_encrypted_network {
        // Definition of the plain network architecture
        println!("WARNING: The encrypted training could require days to be completed");

        let mut model = EncryptedNeuralNetworkU16GPU::create(Some(2));
        model.add_max_pooling(vec![16, 16], 4, 4, 0);
        model.add_conv(1, 1, 2, 2, 2, 0);
        model.add_relu_activation(4);
        model.add_dense(4, 3);

        model.print_plain_weights("Conv2".to_string());
        model.print_plain_biases("Conv2".to_string());
        model.print_plain_weights("Dense4".to_string());
        model.print_plain_biases("Dense4".to_string());

        model.train(
            1,
            1,
            0.1,
            &train_inputs,
            &train_labels,
            vec![1, 1, 16, 16],
            vec![1, 1, 1, 3],
        );

        model.print_plain_weights("Conv2".to_string());
        model.print_plain_biases("Conv2".to_string());
        model.print_plain_weights("Dense4".to_string());
        model.print_plain_biases("Dense4".to_string());
    }

    if test_plain_network {
            
        let mut correct = 0;
        let mut total = val_labels.len();

        for (input, label) in val_inputs.iter().zip(val_labels.iter()) {
            let input_batch = vec![input.clone()]; // batch size = 1
            let input_shape = vec![1, 1, 16, 16];
            let label_shape = vec![1, 1, 1, 3];

            let prediction = plain_model.inference(&input_batch, input_shape.clone(), label_shape.clone());

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

        correct = 0;
        total = test_labels.len();

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
    }

    Ok(())
}