use crate::encrypted_utils::encrypted_context::EncryptedContext;
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::{EncryptedElement, EncryptableValueType};
use crate::encrypted_layers::{EncryptedLayer, EncryptedDenseLayer};
use crate::encrypted_losses::loss_function::LossFunction;

/// Core generic implementation of an encrypted neural network
pub struct EncryptedNeuralNetworkImpl<K: ServerKeyTrait, T: EncryptedElement> {
    layers: Vec<Box<dyn EncryptedLayer<K, T>>>,
    loss: Box<dyn LossFunction<K, T>>,
    context: EncryptedContext<K, T>,
}