use rust_math_tool::core::autodiff::Node;
use rust_math_tool::{Linear, ReLU, MSE, SGD, Layer};
use std::rc::Rc;

fn main() {
    // 1. Setup Data
    let inputs = vec![
        vec![Node::constant(0.0), Node::constant(0.0)],
        vec![Node::constant(0.1), Node::constant(0.9)],
        vec![Node::constant(0.9), Node::constant(0.1)],
        vec![Node::constant(1.0), Node::constant(1.0)],
    ];
    let targets = vec![
        vec![Node::constant(0.0)],
        vec![Node::constant(1.0)],
        vec![Node::constant(1.0)],
        vec![Node::constant(0.0)],
    ];

    // 2. Setup Model
    let l1 = Linear::new(2, 4);
    let relu = ReLU;
    let l2 = Linear::new(4, 1);
    
    let optimizer = SGD::new(0.1);
    let all_params: Vec<Rc<Node>> = [l1.parameters(), l2.parameters()].concat();

    println!("Initial sample weight: {:?}", *all_params[0].value.borrow());
    println!("Starting XOR training...");

    // 3. Training Loop
    for epoch in 0..1000 {
        let mut epoch_loss = 0.0;
        
        for (i, input) in inputs.iter().enumerate() {
            // Forward
            let h1 = l1.forward(input);
            let h1_act = relu.forward(&h1);
            let output = l2.forward(&h1_act);
            
            // Loss
            let loss = MSE::loss(&output, &targets[i]);
            epoch_loss += *loss.value.borrow();
            
            // Backward
            optimizer.zero_grad(&all_params);
            loss.backward();
            
            // Update
            optimizer.step(&all_params);
        }
        
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss {}", epoch, epoch_loss / 4.0);
        }
    }

    println!("Training complete. Final sample weight: {:?}", *all_params[0].value.borrow());
    println!("Final Predictions:");
    for input in inputs {
        let h1 = l1.forward(&input);
        let h1_act = relu.forward(&h1);
        let output = l2.forward(&h1_act);
        println!("Input: {:?}, Predicted: {:?}", 
            [*input[0].value.borrow(), *input[1].value.borrow()],
            *output[0].value.borrow());
    }
}
