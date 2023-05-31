use crate::tflite_flatbuffers::tflite::ActivationFunctionType;
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

#[derive(Copy, Clone)]
pub(crate) enum TokenFusedActivation {
    NONE,
    RELU,
    RELU6,
}

impl ToTokens for TokenFusedActivation {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let output = match self {
            TokenFusedActivation::NONE => quote!(microflow::activation::FusedActivation::NONE),
            TokenFusedActivation::RELU => quote!(microflow::activation::FusedActivation::RELU),
            TokenFusedActivation::RELU6 => quote!(microflow::activation::FusedActivation::RELU6),
        };
        output.to_tokens(tokens);
    }
}

impl From<ActivationFunctionType> for TokenFusedActivation {
    fn from(activation: ActivationFunctionType) -> Self {
        match activation {
            ActivationFunctionType::NONE => Self::NONE,
            ActivationFunctionType::RELU => Self::RELU,
            ActivationFunctionType::RELU6 => Self::RELU6,
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_activation_to_tokens() {
        let activation = TokenFusedActivation::from(ActivationFunctionType::RELU);
        assert_eq!(
            activation.to_token_stream().to_string(),
            quote!(microflow::activation::FusedActivation::RELU).to_string()
        );
    }
}
