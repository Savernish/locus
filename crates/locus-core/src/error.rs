use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum CoreError {
    #[error("polynomial coefficient slice cannot be empty")]
    EmptyCoefficients,
    #[error("denominator polynomial cannot be zero")]
    ZeroDenominator,
    #[error("numerator degree ({num_deg}) exceeds denominator degree ({den_deg}); transfer function is improper")]
    Improper { num_deg: usize, den_deg: usize },
    #[error("division by zero polynomial")]
    DivisionByZero,
}
