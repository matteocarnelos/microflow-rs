use nalgebra::SMatrix;

/// Represents a 2-dimensional buffer.
/// A 2-dimensional buffer is composed by a [`SMatrix`] of values `T`.
pub type Buffer2D<T, const ROWS: usize, const COLUMNS: usize> = SMatrix<T, ROWS, COLUMNS>;

/// Represents a 4-dimensional buffer.
/// A 4-dimensional buffer is composed by an array of [`Buffer2D`] containing an array of values
/// `T`.
pub type Buffer4D<
    T,
    const BATCHES: usize,
    const ROWS: usize,
    const COLUMNS: usize,
    const CHANNELS: usize,
> = [Buffer2D<[T; CHANNELS], ROWS, COLUMNS>; BATCHES];
