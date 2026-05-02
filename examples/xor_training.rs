use rust_math_tool::core::autodiff::Node;
use rust_math_tool::{Linear, ReLU, MSE, SGD, Layer};
use std::rc::Rc;
use ndarray::array;

fn main() {
    // 1. Setup Data (Tensors)
    // XOR inputs as (4, 2) tensor
    let inputs_val = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn();
    let targets_val = array![[0.0], [1.0], [1.0], [0.0]].into_dyn();
    
    let inputs = Node::constant(inputs_val);
    let targets = Node::constant(targets_val);

    // 2. Setup Model
    let l1 = Linear::new(2, 4);
    let relu = ReLU;
    let l2 = Linear::new(4, 1);
    
    let optimizer = SGD::new(0.5);
    let all_params: Vec<Rc<Node>> = [l1.parameters(), l2.parameters()].concat();

    println!("Starting Tensor-Based XOR training...");

    // 3. Training Loop
    for epoch in 0..2000 {
        // Forward
        let h1 = l1.forward(inputs.clone());
        let h1_act = relu.forward(h1);
        let output = l2.forward(h1_act);
        
        // Loss
        let loss = MSE::loss(&output, &targets);
        let loss_val = *loss.value.borrow().iter().next().unwrap();
        
        // Backward
        optimizer.zero_grad(&all_params);
        loss.backward();
        
        // Update
        optimizer.step(&all_params);
        
        if epoch % 200 == 0 {
            println!("Epoch {}: Loss {}", epoch, loss_val);
        }
    }

    // 4. Evaluation
    let h1 = l1.forward(inputs.clone());
    let h1_act = relu.forward(h1);
    let output = l2.forward(h1_act);
    println!("Training complete. Final Predictions:");
    println!("{:?}", *output.value.borrow());
}
