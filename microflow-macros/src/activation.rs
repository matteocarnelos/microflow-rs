use crate::tflite_flatbuffers::tflite::ActivationFunctionType;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};

/// Represents the tokenized version of the [`FusedActivation`].
#[derive(Copy, Clone)]
pub(crate) enum TokenFusedActivation {
    None,
    Relu,
    Relu6,
}

impl ToTokens for TokenFusedActivation {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let ts = match self {
            TokenFusedActivation::None => quote!(microflow::activation::FusedActivation::None),
            TokenFusedActivation::Relu => quote!(microflow::activation::FusedActivation::Relu),
            TokenFusedActivation::Relu6 => quote!(microflow::activation::FusedActivation::Relu6),
        };
        ts.to_tokens(tokens);
    }
}

impl From<ActivationFunctionType> for TokenFusedActivation {
    fn from(activation: ActivationFunctionType) -> Self {
        match activation {
            ActivationFunctionType::NONE => Self::None,
            ActivationFunctionType::RELU => Self::Relu,
            ActivationFunctionType::RELU6 => Self::Relu6,
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
            quote!(microflow::activation::FusedActivation::Relu).to_string()
        );
    }
}
