#!/usr/bin/env python3
"""
Generate synthetic Verilog dataset for fine-tuning.
Combines techniques from CodeV, HaVen, CraftRTL, and VeriReason.
"""

import json
import random
import argparse
from pathlib import Path

# Verilog module templates for synthetic generation
VERILOG_TEMPLATES = [
    {
        "name": "4-bit_adder",
        "spec": "Implement a 4-bit ripple carry adder with two 4-bit inputs A and B, a carry-in Cin, a 4-bit sum output S, and a carry-out Cout.",
        "reasoning": "A ripple carry adder chains full adders. Each bit position needs a full adder that takes A[i], B[i], and the carry from the previous stage. The first stage uses Cin. The carry propagates through each stage to produce Cout.",
        "code": """module adder_4bit (
    input [3:0] A, B,
    input Cin,
    output [3:0] S,
    output Cout
);
    wire [4:0] carry;
    assign carry[0] = Cin;
    
    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : fa
            assign S[i] = A[i] ^ B[i] ^ carry[i];
            assign carry[i+1] = (A[i] & B[i]) | (A[i] & carry[i]) | (B[i] & carry[i]);
        end
    endgenerate
    
    assign Cout = carry[4];
endmodule"""
    },
    {
        "name": "2to1_mux",
        "spec": "Design a 2-to-1 multiplexer with 8-bit data inputs A and B, a select signal Sel, and an 8-bit output Y. When Sel=0, output A; when Sel=1, output B.",
        "reasoning": "A 2-to-1 MUX selects between two inputs based on a select line. The logic is Y = (~Sel & A) | (Sel & B). For 8-bit data, apply this to each bit independently.",
        "code": """module mux2to1_8bit (
    input [7:0] A, B,
    input Sel,
    output [7:0] Y
);
    assign Y = Sel ? B : A;
endmodule"""
    },
    {
        "name": "dff_sync_reset",
        "spec": "Create a D flip-flop with synchronous active-high reset. Inputs: D, clk, rst. Output: Q. On reset, Q becomes 0. Otherwise, Q updates to D on rising clock edge.",
        "reasoning": "A D flip-flop stores its input on the clock edge. With synchronous reset, the reset is checked on the clock edge, not asynchronously. Use always_ff block for synthesis.",
        "code": """module dff_sync_reset (
    input D, clk, rst,
    output reg Q
);
    always_ff @(posedge clk) begin
        if (rst)
            Q <= 1'b0;
        else
            Q <= D;
    end
endmodule"""
    },
    {
        "name": "counter_8bit",
        "spec": "Implement an 8-bit up-counter with synchronous reset and enable. When enable is high, count increments on each clock rising edge. Reset clears count to 0.",
        "reasoning": "A counter is a register that increments. Synchronous reset clears on clock edge. Enable gate controls whether increment happens. Use non-blocking assignment for sequential logic.",
        "code": """module counter_8bit (
    input clk, rst, en,
    output reg [7:0] count
);
    always_ff @(posedge clk) begin
        if (rst)
            count <= 8'd0;
        else if (en)
            count <= count + 8'd1;
    end
endmodule"""
    },
    {
        "name": "decoder_3to8",
        "spec": "Design a 3-to-8 decoder with a 3-bit input In and an 8-bit output Out. Only the output bit corresponding to the binary value of In should be high.",
        "reasoning": "A decoder activates one of 2^N outputs based on N inputs. For 3-to-8, output bit In should be 1, all others 0. Use left shift: Out = 1 << In.",
        "code": """module decoder_3to8 (
    input [2:0] In,
    output [7:0] Out
);
    assign Out = 8'b1 << In;
endmodule"""
    },
    {
        "name": "priority_encoder",
        "spec": "Create an 8-input priority encoder. Inputs: in[7:0]. Outputs: out[2:0] (encoded value of highest priority input), valid (high if any input is high). Priority: in[7] highest, in[0] lowest.",
        "reasoning": "Priority encoder finds the index of the highest-order 1 bit. Use casex or if-else chain. valid is OR reduction of inputs. Output is index of first 1 from MSB.",
        "code": """module priority_encoder_8to3 (
    input [7:0] in,
    output reg [2:0] out,
    output valid
);
    assign valid = |in;
    
    always_comb begin
        if (in[7]) out = 3'd7;
        else if (in[6]) out = 3'd6;
        else if (in[5]) out = 3'd5;
        else if (in[4]) out = 3'd4;
        else if (in[3]) out = 3'd3;
        else if (in[2]) out = 3'd2;
        else if (in[1]) out = 3'd1;
        else out = 3'd0;
    end
endmodule"""
    },
    {
        "name": "shift_register",
        "spec": "Implement a 4-bit shift register with parallel load, serial in, and left shift. Controls: clk, rst, load (parallel load), shift (shift left), serial_in. Parallel input: din[3:0]. Output: q[3:0].",
        "reasoning": "Shift register can either load in parallel or shift left. When shifting, MSB takes serial_in, others shift up. Use case statement or if-else for control.",
        "code": """module shift_register_4bit (
    input clk, rst, load, shift,
    input serial_in,
    input [3:0] din,
    output reg [3:0] q
);
    always_ff @(posedge clk) begin
        if (rst)
            q <= 4'b0;
        else if (load)
            q <= din;
        else if (shift)
            q <= {q[2:0], serial_in};
    end
endmodule"""
    },
    {
        "name": "comparator",
        "spec": "Design an 8-bit comparator with inputs A and B. Outputs: A_gt_B (A > B), A_eq_B (A == B), A_lt_B (A < B).",
        "reasoning": "Compare two unsigned numbers. Greater than, equal, less than are mutually exclusive. Use standard comparison operators which synthesize to subtractor + zero detect.",
        "code": """module comparator_8bit (
    input [7:0] A, B,
    output A_gt_B, A_eq_B, A_lt_B
);
    assign A_gt_B = (A > B);
    assign A_eq_B = (A == B);
    assign A_lt_B = (A < B);
endmodule"""
    },
    {
        "name": "fsm_traffic_light",
        "spec": "Design a traffic light controller FSM with states: RED (2 cycles), GREEN (2 cycles), YELLOW (1 cycle). Input: clk, rst. Outputs: red, green, yellow. Use a 2-bit counter for timing within each state.",
        "reasoning": "Use enum for states. Counter increments each cycle. When counter reaches limit, transition to next state. Reset to RED with counter=0.",
        "code": """module traffic_light_fsm (
    input clk, rst,
    output reg red, green, yellow
);
    typedef enum logic [1:0] {RED, GREEN, YELLOW} state_t;
    state_t state, next_state;
    reg [1:0] timer;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= RED;
            timer <= 2'd0;
        end else begin
            state <= next_state;
            if (state != next_state)
                timer <= 2'd0;
            else
                timer <= timer + 2'd1;
        end
    end
    
    always_comb begin
        next_state = state;
        case (state)
            RED:   if (timer == 2'd1) next_state = GREEN;
            GREEN: if (timer == 2'd1) next_state = YELLOW;
            YELLOW: if (timer == 2'd0) next_state = RED;
        endcase
    end
    
    always_comb begin
        red = (state == RED);
        green = (state == GREEN);
        yellow = (state == YELLOW);
    end
endmodule"""
    },
    {
        "name": "bcd_to_7seg",
        "spec": "Design a BCD to 7-segment decoder. Input: bcd[3:0] (0-9). Output: seg[6:0] (a-g segments, active high).",
        "reasoning": "7-segment display has segments a-g. Each digit 0-9 lights specific segments. Use case statement mapping BCD input to 7-bit segment pattern.",
        "code": """module bcd_to_7seg (
    input [3:0] bcd,
    output reg [6:0] seg  // a,b,c,d,e,f,g
);
    always_comb begin
        case (bcd)
            4'd0: seg = 7'b1111110;
            4'd1: seg = 7'b0110000;
            4'd2: seg = 7'b1101101;
            4'd3: seg = 7'b1111001;
            4'd4: seg = 7'b0110011;
            4'd5: seg = 7'b1011011;
            4'd6: seg = 7'b1011111;
            4'd7: seg = 7'b1110000;
            4'd8: seg = 7'b1111111;
            4'd9: seg = 7'b1111011;
            default: seg = 7'b0000000;
        endcase
    end
endmodule"""
    }
]


def generate_variation(template: dict, idx: int) -> dict:
    """Generate a slightly varied example from template."""
    # Add minor variations to spec
    variations = [
        template["spec"],
        f"Design a Verilog module: {template['spec']}",
        f"Implement the following circuit in Verilog. {template['spec']}",
        f"Write synthesizable Verilog for: {template['spec']}",
    ]
    
    spec = random.choice(variations)
    
    return {
        "spec": spec,
        "code": template["code"],
        "reasoning": template.get("reasoning", ""),
        "source": f"synthetic_{template['name']}",
        "id": f"{template['name']}_{idx}"
    }


def generate_dataset(num_examples: int = 1000, seed: int = 42) -> list:
    """Generate synthetic Verilog dataset."""
    random.seed(seed)
    dataset = []
    
    for i in range(num_examples):
        template = random.choice(VERILOG_TEMPLATES)
        example = generate_variation(template, i)
        dataset.append(example)
    
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/synthetic_train.jsonl")
    parser.add_argument("--num", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print(f"Generating {args.num} synthetic examples...")
    dataset = generate_dataset(args.num, args.seed)
    
    # Split train/eval (90/10)
    split = int(0.9 * len(dataset))
    train = dataset[:split]
    eval_data = dataset[split:]
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    train_path = args.output
    eval_path = args.output.replace("train", "eval")
    
    with open(train_path, 'w') as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    with open(eval_path, 'w') as f:
        for ex in eval_data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"Train: {len(train)} examples -> {train_path}")
    print(f"Eval: {len(eval_data)} examples -> {eval_path}")


if __name__ == "__main__":
    main()
