use locus_core::*;

#[test]
fn series_then_unity_feedback() {
    // G1 = 1/(s+1), G2 = 1/(s+2), in series: 1/(s^2 + 3s + 2)
    // Then unity negative feedback: 1/(s^2 + 3s + 3)
    let g1 = TransferFunction::from_coeffs(vec![1.0], vec![1.0, 1.0]).unwrap();
    let g2 = TransferFunction::from_coeffs(vec![1.0], vec![1.0, 2.0]).unwrap();
    let open = series(&g1, &g2).unwrap();
    let unity = TransferFunction::from_coeffs(vec![1.0], vec![1.0]).unwrap();
    let cl = feedback(&open, &unity).unwrap();
    assert_eq!(cl.num().coeffs(), &[1.0]);
    assert_eq!(cl.den().coeffs(), &[1.0, 3.0, 3.0]);
}

#[test]
fn polynomial_division_roundtrip_through_tf() {
    // Build (s^2 + 3s + 2), divide by (s + 1), expect quotient (s + 2)
    // and verify constructing TF (s + 2)/(s + 1) is valid.
    let n = Polynomial::new(vec![1.0, 3.0, 2.0]).unwrap();
    let d = Polynomial::new(vec![1.0, 1.0]).unwrap();
    let (q, r) = poly_div(&n, &d).unwrap();
    assert_eq!(q.coeffs(), &[1.0, 2.0]);
    assert!(r.is_zero());

    let g = TransferFunction::new(q, d).unwrap();
    assert_eq!(g.order(), 1);
}
