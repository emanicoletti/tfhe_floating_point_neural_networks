use crate::plain_nn_builder::plain_layers::PlainLayer;
use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;

use rayon::iter::*;

pub struct PlainReLUActivation<T: PlainElement> {
    pub id: String,
    pub derivatives: PlainTensor<T>, // stores the activation values
}

impl<T: PlainElement> PlainReLUActivation<T> {
    pub fn new(id: String, derivatives: PlainTensor<T>) -> Self {
        Self { id, derivatives }
    }
}

impl<T> PlainLayer<T> for PlainReLUActivation<T>
where
    T: PlainReLU + PlainBackwardReLU + Send + Sync + Clone + PlainElement,
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let (activations, derivatives) = input.relu();
        self.derivatives = derivatives;
        activations
    }

    fn backward(
        &mut self,
        _input: &PlainTensor<T>,
        grad_output: &PlainTensor<T>,
    ) -> PlainTensor<T> {
        let grad_input_data: Vec<T> = grad_output
            .data
            .par_iter()
            .zip(self.derivatives.data.par_iter())
            .map(|(g, d)| d.clone().backward_relu(g.clone()))
            .collect();

        PlainTensor::new(grad_input_data, grad_output.shape.clone())
    }

    fn update_parameters(&mut self, _learning_rate: T, _weight_decay: T, _momentum: T) {
        // No parameters to update in relu activation
    }

    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        self.forward(input)
    }

    fn exact_inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        self.forward(input)
    }

    fn get_biases(&self) -> PlainTensor<T> {
        // No biases in relu activation
        PlainTensor::new(vec![], vec![])
    }

    fn get_grad_biases(&self) -> PlainTensor<T> {
        // No gradients for biases in relu activation
        PlainTensor::new(vec![], vec![])
    }

    fn get_grad_weights(&self) -> PlainTensor<T> {
        // No gradients for weights in relu activation
        PlainTensor::new(vec![], vec![])
    }

    fn get_weights(&self) -> PlainTensor<T> {
        // No weights in relu activation
        PlainTensor::new(vec![], vec![])
    }

    fn get_id(&self) -> String {
        self.id.clone()
    }
}
