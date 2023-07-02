use crate::tflite_flatbuffers::tflite::{Operator, Tensor};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};

/// Represents the tokenized version of the `Reshape` operator.
pub(crate) struct TokenReshape {
    pub(crate) output_shape: Vec<usize>,
}

/// Parses the [`TokenReshape`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
///
pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    Box::new(TokenReshape::new(operator, tensors))
}

impl TokenReshape {
    /// Builds the [`TokenReshape`] operator from the given model operator and tensors.
    ///
    /// # Arguments
    /// * `operator` - The model operator as an [`Operator`]
    /// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
    ///
    pub(crate) fn new(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>) -> Self {
        let output_shape: Vec<_> = tensors
            .get(operator.outputs().unwrap().get(0) as usize)
            .shape()
            .unwrap()
            .iter()
            .map(|e| e as usize)
            .collect();
        Self { output_shape }
    }
}

impl ToTokens for TokenReshape {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_shape = &self.output_shape;
        let output_tensor = match output_shape.len() {
            2 => quote!(Tensor2D),
            4 => quote!(Tensor4D),
            _ => unimplemented!(),
        };

        let ts = quote! {
            let input: microflow::tensor::#output_tensor<_, #(#output_shape),*, 1usize> =
                microflow::ops::reshape(input);
        };
        ts.to_tokens(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> TokenReshape {
        TokenReshape {
            output_shape: vec![2, 3],
        }
    }

    #[test]
    fn reshape_to_tokens() {
        let layer = setup();
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor2D<_, 2usize, 3usize, 1usize> =
                    microflow::ops::reshape(input);
            }
            .to_string()
        )
    }
}
