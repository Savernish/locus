use num_complex::Complex64;
use thiserror::Error;

pub type Time = f64;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("denominator polynomial cannot be empty")]
    EmptyDenominator,
    #[error("numerator degree ({num}) exceeds denominator degree ({den}); system is improper")]
    Improper { num: usize, den: usize },
}

#[derive(Debug, Clone)]
pub struct TransferFunction {
    pub num: Vec<f64>,
    pub den: Vec<f64>,
}

impl TransferFunction {
    pub fn new(num: Vec<f64>, den: Vec<f64>) -> Result<Self, CoreError> {
        if den.is_empty() {
            return Err(CoreError::EmptyDenominator);
        }
        if num.len() > den.len() {
            return Err(CoreError::Improper {
                num: num.len() - 1,
                den: den.len() - 1,
            });
        }
        Ok(Self { num, den })
    }

    pub fn order(&self) -> usize {
        self.den.len().saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_order_constructs() {
        let g = TransferFunction::new(vec![1.0], vec![1.0, 2.0, 1.0]).unwrap();
        assert_eq!(g.order(), 2);
    }

    #[test]
    fn improper_rejected() {
        let r = TransferFunction::new(vec![1.0, 1.0, 1.0], vec![1.0, 1.0]);
        assert!(matches!(r, Err(CoreError::Improper { .. })));
    }

    #[test]
    fn empty_den_rejected() {
        let r = TransferFunction::new(vec![1.0], vec![]);
        assert!(matches!(r, Err(CoreError::EmptyDenominator)));
    }
}
