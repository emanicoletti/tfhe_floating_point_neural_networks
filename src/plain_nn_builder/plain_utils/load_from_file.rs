use ndarray::Array2;
use ndarray_npy::read_npy;
use std::path::Path;
use std::error::Error;

pub fn array2_to_vecvec(array: &Array2<f32>) -> Vec<Vec<f32>> {
    array
        .outer_iter() // iterates over rows
        .map(|row| row.to_vec())
        .collect()
}


pub fn load_data(exp_number: i8) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>), Box<dyn Error>> {
    let (x_train, y_train, x_val, y_val, x_test, y_test) = match exp_number {
        1 | 2 => {
            let x_train: Array2<f32> = read_npy(Path::new("src/experiment_1_2/dataset/x_train.npy"))?;
            let y_train: Array2<f32> = read_npy(Path::new("src/experiment_1_2/dataset/y_train.npy"))?;
            let x_val: Array2<f32> = read_npy(Path::new("src/experiment_1_2/dataset/x_val.npy"))?;
            let y_val: Array2<f32> = read_npy(Path::new("src/experiment_1_2/dataset/y_val.npy"))?;
            let x_test: Array2<f32> = read_npy(Path::new("src/experiment_1_2/dataset/x_test.npy"))?;
            let y_test: Array2<f32> = read_npy(Path::new("src/experiment_1_2/dataset/y_test.npy"))?;
            (x_train, y_train, x_val, y_val, x_test, y_test)
        },
        3 => {
            let x_train: Array2<f32> = read_npy(Path::new("src/experiment_3/dataset/x_train.npy"))?;
            let y_train: Array2<f32> = read_npy(Path::new("src/experiment_3/dataset/y_train.npy"))?;
            let x_test: Array2<f32> = read_npy(Path::new("src/experiment_3/dataset/x_test.npy"))?;
            let y_test: Array2<f32> = read_npy(Path::new("src/experiment_3/dataset/y_test.npy"))?;
            let empty_array = Array2::<f32>::zeros((0, 0));
            (x_train, y_train, empty_array.clone(), empty_array.clone(), x_test, y_test)
        }
        _ => return Err("Invalid experiment number".into()),
    };

    Ok((x_train, y_train, x_val, y_val, x_test, y_test))
}
