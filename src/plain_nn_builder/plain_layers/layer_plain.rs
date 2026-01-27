use crate::plain_nn_builder::plain_utils::{PlainTensor, PlainElement};

pub trait PlainLayer<T>: Send + Sync
where
    T: PlainElement,
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T>;

    fn backward(
        &mut self,
        input: &PlainTensor<T>,
        grad_output: &PlainTensor<T>,
    ) -> PlainTensor<T>
    where T: Default;

    fn update_parameters(
        &mut self,
        learning_rate: T,
        weight_decay: T,
        momentum: T,
    );

    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T>;

    #[allow(dead_code)]
    // Use the approximate arithmetic for inference, while keeping training with exact arithmetic (Experimental)
    fn approximate_inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T>;

    fn get_weights(&self) -> PlainTensor<T>;
    
    fn get_biases(&self) -> PlainTensor<T>;

    #[allow(dead_code)]
    // Debugging purposes
    fn get_grad_weights(&self) -> PlainTensor<T>;

    #[allow(dead_code)]
    // Debugging purposes
    fn get_grad_biases(&self) -> PlainTensor<T>;

    fn get_id(&self) -> String;

}