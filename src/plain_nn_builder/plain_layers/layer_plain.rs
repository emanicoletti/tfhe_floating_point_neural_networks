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
        learning_rate: T
    );

    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T>;

    fn get_weights(&self) -> PlainTensor<T>;
    
    fn get_biases(&self) -> PlainTensor<T>;

    fn get_grad_weights(&self) -> PlainTensor<T>;

    fn get_grad_biases(&self) -> PlainTensor<T>;

    fn get_id(&self) -> String;
}