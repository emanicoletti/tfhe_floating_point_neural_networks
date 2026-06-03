use crate::plain_nn_builder::plain_layers::PlainLayer;
use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;

pub struct GlobalAveragePooling<T>
where
    T: PlainElement,
{
    id: String,
    _marker: std::marker::PhantomData<T>,
}

impl<T> GlobalAveragePooling<T>
where
    T: PlainElement,
{
    pub fn new(id: String) -> Self {
        Self {
            id,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> PlainLayer<T> for GlobalAveragePooling<T>
where
    T: PlainElement + PlainMul + PlainAdd + PlainDiv + PlainValueType + Default,
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let shape = input.clone().shape;
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let spatial_dim = height * width;
        let scale_factor = T::from_f32(1.0 / (spatial_dim as f32));

        let mut output_data = Vec::with_capacity(batch_size * channels);

        for feature_map in input.data.chunks(spatial_dim) {
            let sum: T = feature_map
                .iter()
                .fold(T::default(), |acc, x| acc.add(x.clone()));
            let avg = sum.mul(scale_factor.clone());
            output_data.push(avg);
        }

        PlainTensor::new(output_data, vec![batch_size, channels, 1, 1])
    }

    fn backward(&mut self, input: &PlainTensor<T>, grad_output: &PlainTensor<T>) -> PlainTensor<T> {
        let shape = input.clone().shape;
        let spatial_dim = shape[2] * shape[3];
        let scale_factor = T::from_f32(1.0 / (spatial_dim as f32));

        let mut grad_input_data = Vec::with_capacity(input.data.len());

        for grad_val in grad_output.data.iter() {
            let distributed_grad = grad_val.clone().mul(scale_factor.clone());

            for _ in 0..spatial_dim {
                grad_input_data.push(distributed_grad.clone());
            }
        }

        PlainTensor::new(grad_input_data, shape.clone())
    }

    fn update_parameters(&mut self, _learning_rate: T, _weight_decay: T, _momentum: T) {}

    fn get_id(&self) -> String {
        self.id.clone()
    }
    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        self.forward(input)
    }
    fn exact_inference(&mut self, _input: &PlainTensor<T>) -> PlainTensor<T> {
        todo!()
    }
    fn get_weights(&self) -> PlainTensor<T> {
        PlainTensor::new(vec![], vec![])
    }
    fn get_biases(&self) -> PlainTensor<T> {
        PlainTensor::new(vec![], vec![])
    }
    fn get_grad_weights(&self) -> PlainTensor<T> {
        PlainTensor::new(vec![], vec![])
    }
    fn get_grad_biases(&self) -> PlainTensor<T> {
        PlainTensor::new(vec![], vec![])
    }
}
