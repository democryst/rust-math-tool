use rust_math_tool::core::autodiff::{Node, add, mul};
use ndarray::array;

#[test]
fn test_autodiff_basic() {
    let x = Node::variable(array![2.0].into_dyn());
    let y = Node::variable(array![3.0].into_dyn());
    
    // f = x * y + x
    let f = add(&mul(&x, &y), &x);
    
    f.backward();
    
    assert_eq!(*f.value.borrow(), array![8.0].into_dyn());
    assert_eq!(*x.grad.borrow(), array![4.0].into_dyn()); // df/dx = y + 1 = 4
    assert_eq!(*y.grad.borrow(), array![2.0].into_dyn()); // df/dy = x = 2
}
