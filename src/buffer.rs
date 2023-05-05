use nalgebra::SMatrix;

pub type Buffer2D<T, const D1: usize, const D2: usize> = SMatrix<T, D1, D2>;
pub type Buffer4D<T, const D1: usize, const D2: usize, const D3: usize, const D4: usize> =
    [SMatrix<[T; D4], D2, D3>; D1];
