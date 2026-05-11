"""
Comprehensive Verilog module templates with SI-CoT reasoning.
Covers combinational, sequential, arithmetic, memory, and interface circuits.
Each template includes spec, reasoning (HaVen-style), and compile-validated code.
"""

VERILOG_TEMPLATES = [
    # ============ COMBINATIONAL ============
    {
        "name": "mux2to1",
        "category": "combinational",
        "spec": "Design a parameterized N-bit 2-to-1 multiplexer. Inputs: A[N-1:0], B[N-1:0], Sel. Output: Y[N-1:0]. When Sel=0, Y=A; when Sel=1, Y=B. Use parameter N=8.",
        "reasoning": "A 2-to-1 MUX selects between two N-bit inputs based on Sel. The ternary operator (Sel ? B : A) works vector-wise in Verilog, so a single assign statement handles all N bits. Using a parameter makes it reusable for any width.",
        "code": """module mux2to1 #(parameter N = 8) (
    input  [N-1:0] A,
    input  [N-1:0] B,
    input          Sel,
    output [N-1:0] Y
);
    assign Y = Sel ? B : A;
endmodule"""
    },
    {
        "name": "mux4to1",
        "category": "combinational",
        "spec": "Design a 4-to-1 multiplexer with 8-bit data inputs A, B, C, D and 2-bit select Sel. Output Y. Sel selects which input passes to output.",
        "reasoning": "4-to-1 MUX needs 2 select bits (2^2=4). Use case statement or nested ternary. Case is cleaner and synthesizes to same logic. Each case value selects one input.",
        "code": """module mux4to1 (
    input  [7:0] A, B, C, D,
    input  [1:0] Sel,
    output reg [7:0] Y
);
    always_comb begin
        case (Sel)
            2'b00: Y = A;
            2'b01: Y = B;
            2'b10: Y = C;
            2'b11: Y = D;
            default: Y = 8'b0;
        endcase
    end
endmodule"""
    },
    {
        "name": "decoder_3to8",
        "category": "combinational",
        "spec": "Design a 3-to-8 decoder. Input In[2:0], output Out[7:0]. Only the output bit at position In is high. Include enable signal En (active high). When En=0, all outputs low.",
        "reasoning": "Decoder activates one of 2^N outputs. With enable, output is conditional. Use left shift: 1 << In gives a one-hot vector. When disabled, output is 0. This is more compact than case statement and synthesizes efficiently.",
        "code": """module decoder_3to8 (
    input  [2:0] In,
    input        En,
    output [7:0] Out
);
    assign Out = En ? (8'b1 << In) : 8'b0;
endmodule"""
    },
    {
        "name": "encoder_8to3",
        "category": "combinational",
        "spec": "Design an 8-to-3 priority encoder. Input in[7:0], output out[2:0] (encoded index), valid (high if any input is high). Priority: in[7] is highest, in[0] lowest.",
        "reasoning": "Priority encoder finds the highest-index 1 bit. Use if-else chain from MSB to LSB. valid is reduction-OR of inputs. The first true condition in the chain sets the output, giving natural priority.",
        "code": """module priority_encoder_8to3 (
    input  [7:0] in,
    output reg [2:0] out,
    output       valid
);
    assign valid = |in;
    always_comb begin
        if      (in[7]) out = 3'd7;
        else if (in[6]) out = 3'd6;
        else if (in[5]) out = 3'd5;
        else if (in[4]) out = 3'd4;
        else if (in[3]) out = 3'd3;
        else if (in[2]) out = 3'd2;
        else if (in[1]) out = 3'd1;
        else            out = 3'd0;
    end
endmodule"""
    },
    {
        "name": "comparator",
        "category": "combinational",
        "spec": "Design an 8-bit unsigned comparator with inputs A, B. Outputs: A_gt_B, A_eq_B, A_lt_B. Also output difference A_minus_B[8:0] for debugging.",
        "reasoning": "Three comparisons are mutually exclusive. Use standard operators which synthesize to subtractor + zero/sign detection. A_minus_B uses 9 bits to capture borrow for underflow detection.",
        "code": """module comparator_8bit (
    input  [7:0] A, B,
    output       A_gt_B,
    output       A_eq_B,
    output       A_lt_B,
    output [8:0] A_minus_B
);
    assign A_gt_B    = (A > B);
    assign A_eq_B    = (A == B);
    assign A_lt_B    = (A < B);
    assign A_minus_B = {1'b0, A} - {1'b0, B};
endmodule"""
    },
    {
        "name": "alu_4bit",
        "category": "combinational",
        "spec": "Design a 4-bit ALU with operations: add (000), sub (001), AND (010), OR (011), XOR (100), NOT A (101). Inputs: A[3:0], B[3:0], op[2:0]. Outputs: result[3:0], zero, carry.",
        "reasoning": "ALU uses case statement to select operation. Arithmetic ops need carry tracking, so use 5-bit intermediate. Logical ops don't affect carry. Zero flag checks if result is 0.",
        "code": """module alu_4bit (
    input  [3:0] A, B,
    input  [2:0] op,
    output reg [3:0] result,
    output reg       carry,
    output           zero
);
    reg [4:0] tmp;
    always_comb begin
        carry = 1'b0;
        case (op)
            3'b000: begin tmp = A + B; result = tmp[3:0]; carry = tmp[4]; end
            3'b001: begin tmp = A - B; result = tmp[3:0]; carry = tmp[4]; end
            3'b010: begin result = A & B; end
            3'b011: begin result = A | B; end
            3'b100: begin result = A ^ B; end
            3'b101: begin result = ~A; end
            default: begin result = 4'b0; end
        endcase
    end
    assign zero = (result == 4'b0);
endmodule"""
    },
    {
        "name": "bcd_to_7seg",
        "category": "combinational",
        "spec": "Design a BCD to 7-segment decoder. Input bcd[3:0] (0-9). Output seg[6:0] where seg[0]=a, seg[1]=b, ..., seg[6]=g, active high. Invalid BCD (10-15) turns off all segments.",
        "reasoning": "7-segment display lights specific segments per digit. Each BCD value maps to a 7-bit pattern. Use case statement for clean mapping. Default handles invalid BCD by turning off all segments.",
        "code": """module bcd_to_7seg (
    input  [3:0] bcd,
    output reg [6:0] seg
);
    always_comb begin
        case (bcd)
            4'd0: seg = 7'b1111110; // a,b,c,d,e,f
            4'd1: seg = 7'b0110000; // b,c
            4'd2: seg = 7'b1101101; // a,b,d,e,g
            4'd3: seg = 7'b1111001; // a,b,c,d,g
            4'd4: seg = 7'b0110011; // b,c,f,g
            4'd5: seg = 7'b1011011; // a,c,d,f,g
            4'd6: seg = 7'b1011111; // a,c,d,e,f,g
            4'd7: seg = 7'b1110000; // a,b,c
            4'd8: seg = 7'b1111111; // all
            4'd9: seg = 7'b1111011; // a,b,c,d,f,g
            default: seg = 7'b0000000;
        endcase
    end
endmodule"""
    },
    {
        "name": "barrel_shifter",
        "category": "combinational",
        "spec": "Design an 8-bit barrel shifter. Inputs: data[7:0], shift[2:0] (shift amount), dir (0=right, 1=left). Output: result[7:0]. Logical shift (fill with 0).",
        "reasoning": "Barrel shifter shifts by variable amount in one clock level (not iterative). For left shift: result = data << shift. For right shift: result = data >> shift. Use mux or conditional assign. Verilog's built-in shift operators synthesize to barrel shifter directly.",
        "code": """module barrel_shifter_8bit (
    input  [7:0] data,
    input  [2:0] shift,
    input        dir,      // 0=right, 1=left
    output [7:0] result
);
    assign result = dir ? (data << shift) : (data >> shift);
endmodule"""
    },
    {
        "name": "parity_generator",
        "category": "combinational",
        "spec": "Design an 8-bit even parity generator. Input data[7:0]. Output parity bit that makes total number of 1s (data + parity) even.",
        "reasoning": "Even parity bit is XOR of all data bits. If data has odd number of 1s, parity=1 to make total even. XOR reduction (^data) computes this in one operation.",
        "code": """module parity_generator (
    input  [7:0] data,
    output       parity
);
    assign parity = ^data;  // XOR reduction
endmodule"""
    },
    {
        "name": "majority_voter",
        "category": "combinational",
        "spec": "Design a 5-input majority voter. Inputs a,b,c,d,e. Output y is 1 if 3 or more inputs are 1.",
        "reasoning": "Count the number of 1s among 5 inputs. If sum >= 3, output is 1. Sum all inputs as a 3-bit value, then compare to 3. This is cleaner than enumerating all majority combinations.",
        "code": """module majority_voter (
    input a, b, c, d, e,
    output y
);
    assign y = (a + b + c + d + e) >= 3'd3;
endmodule"""
    },
    # ============ SEQUENTIAL ============
    {
        "name": "dff_sync_reset",
        "category": "sequential",
        "spec": "Design a D flip-flop with synchronous active-high reset. Inputs: D, clk, rst. Output: Q. On rst=1, Q=0. Otherwise Q updates to D on rising clock edge.",
        "reasoning": "Synchronous means reset is checked on clock edge, not asynchronously. Use always_ff with posedge clk sensitivity only. Non-blocking assignment (<=) for sequential logic.",
        "code": """module dff_sync_reset (
    input  D,
    input  clk,
    input  rst,
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
        "name": "dff_async_reset",
        "category": "sequential",
        "spec": "Design a D flip-flop with asynchronous active-low reset. Inputs: D, clk, rst_n. Output: Q. rst_n takes effect immediately (not just on clock edge).",
        "reasoning": "Asynchronous reset responds immediately, so it goes in the sensitivity list. Active-low means reset when rst_n=0. Use always_ff with both posedge clk and negedge rst_n.",
        "code": """module dff_async_reset (
    input  D,
    input  clk,
    input  rst_n,
    output reg Q
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            Q <= 1'b0;
        else
            Q <= D;
    end
endmodule"""
    },
    {
        "name": "counter_8bit_up",
        "category": "sequential",
        "spec": "Design an 8-bit up-counter with synchronous active-high reset and count enable. When en=1, count increments on rising clock edge. When rst=1, count clears to 0.",
        "reasoning": "Counter is a register that increments conditionally. Synchronous reset clears on clock edge. Enable gate controls whether increment happens. Non-blocking assignment for sequential.",
        "code": """module counter_8bit_up (
    input        clk,
    input        rst,
    input        en,
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
        "name": "counter_updown",
        "category": "sequential",
        "spec": "Design a 4-bit up/down counter. Inputs: clk, rst, en, up (1=up, 0=down). Output count[3:0]. When up=1 and en=1, count++. When up=0 and en=1, count--. rst clears to 0.",
        "reasoning": "Up/down counter selects between increment and decrement based on direction signal. Same reset/enable logic as simple counter, but with conditional increment/decrement.",
        "code": """module counter_updown_4bit (
    input        clk,
    input        rst,
    input        en,
    input        up,
    output reg [3:0] count
);
    always_ff @(posedge clk) begin
        if (rst)
            count <= 4'd0;
        else if (en) begin
            if (up)
                count <= count + 4'd1;
            else
                count <= count - 4'd1;
        end
    end
endmodule"""
    },
    {
        "name": "shift_register_piso",
        "category": "sequential",
        "spec": "Design a 4-bit parallel-in serial-out (PISO) shift register. Inputs: clk, rst, load, parallel_in[3:0]. Output: serial_out. When load=1, load parallel_in. Otherwise shift left on each clock, outputting MSB.",
        "reasoning": "PISO loads all bits in parallel, then shifts them out one at a time. Shift left means next MSB becomes output. The register is {reg[2:0], 1'b0} for left shift, or concatenate with 0 on right.",
        "code": """module shift_register_piso (
    input        clk,
    input        rst,
    input        load,
    input  [3:0] parallel_in,
    output       serial_out
);
    reg [3:0] reg_data;
    
    always_ff @(posedge clk) begin
        if (rst)
            reg_data <= 4'b0;
        else if (load)
            reg_data <= parallel_in;
        else
            reg_data <= {reg_data[2:0], 1'b0};
    end
    
    assign serial_out = reg_data[3];
endmodule"""
    },
    {
        "name": "lfsr_4bit",
        "category": "sequential",
        "spec": "Design a 4-bit Linear Feedback Shift Register (LFSR) with taps at positions 4 and 1 (polynomial x^4 + x + 1). Input: clk, rst, en. Output: out[3:0]. On rst, seed with 4'b0001. On en, shift and feedback XOR of tap positions.",
        "reasoning": "LFSR generates pseudo-random sequences. Feedback is XOR of tap bits. On each clock, shift left and inject feedback into LSB. Must avoid all-zeros state (stuck state). Seed with non-zero value.",
        "code": """module lfsr_4bit (
    input        clk,
    input        rst,
    input        en,
    output [3:0] out
);
    reg [3:0] state;
    wire feedback = state[3] ^ state[0];
    
    always_ff @(posedge clk) begin
        if (rst)
            state <= 4'b0001;
        else if (en)
            state <= {state[2:0], feedback};
    end
    
    assign out = state;
endmodule"""
    },
    {
        "name": "edge_detector",
        "category": "sequential",
        "spec": "Design a rising edge detector. Input: clk, rst, signal. Output: pulse (one clock cycle high when signal rises from 0 to 1).",
        "reasoning": "Edge detection requires remembering previous value. Delay signal by one clock. Rising edge = current AND (NOT previous). Register previous value each cycle.",
        "code": """module edge_detector (
    input  clk,
    input  rst,
    input  signal,
    output pulse
);
    reg sig_d;
    always_ff @(posedge clk) begin
        if (rst)
            sig_d <= 1'b0;
        else
            sig_d <= signal;
    end
    assign pulse = signal & ~sig_d;
endmodule"""
    },
    {
        "name": "debouncer",
        "category": "sequential",
        "spec": "Design a switch debouncer. Inputs: clk, rst, noisy_in. Output: clean_out. Use a 4-bit counter. When noisy_in is stable for 16 cycles, update clean_out. Reset clears to 0.",
        "reasoning": "Mechanical switches bounce, producing multiple transitions. Debouncer waits for input to be stable for several cycles before updating output. A counter increments when input equals output candidate, resets on mismatch. When counter saturates, update output.",
        "code": """module debouncer (
    input  clk,
    input  rst,
    input  noisy_in,
    output reg clean_out
);
    reg [3:0] counter;
    reg       stable;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            counter    <= 4'd0;
            clean_out  <= 1'b0;
        end else begin
            if (noisy_in == clean_out) begin
                counter <= 4'd0;
            end else begin
                if (counter == 4'd15) begin
                    clean_out <= noisy_in;
                    counter   <= 4'd0;
                end else begin
                    counter <= counter + 4'd1;
                end
            end
        end
    end
endmodule"""
    },
    {
        "name": "pwm_generator",
        "category": "sequential",
        "spec": "Design a PWM generator. Inputs: clk, rst, duty[7:0] (0-255). Output: pwm. Counter runs 0-255. When counter < duty, pwm=1, else 0. This gives duty/256 duty cycle.",
        "reasoning": "PWM compares a free-running counter to a duty value. Counter increments every cycle. Output is high when counter < duty. Frequency is clk/256. Duty cycle = duty/256.",
        "code": """module pwm_generator (
    input        clk,
    input        rst,
    input  [7:0] duty,
    output       pwm
);
    reg [7:0] counter;
    always_ff @(posedge clk) begin
        if (rst)
            counter <= 8'd0;
        else
            counter <= counter + 8'd1;
    end
    assign pwm = (counter < duty);
endmodule"""
    },
    {
        "name": "frequency_divider",
        "category": "sequential",
        "spec": "Design a clock frequency divider by 10. Input: clk, rst. Output: div_clk. Toggle output every 5 input cycles (so output period is 10 input cycles).",
        "reasoning": "Divide-by-10 means output toggles every 5 cycles (half period). Count 0-4, then toggle output and reset counter. This produces 50% duty cycle divided clock.",
        "code": """module freq_divider_10 (
    input  clk,
    input  rst,
    output reg div_clk
);
    reg [3:0] counter;
    always_ff @(posedge clk) begin
        if (rst) begin
            counter <= 4'd0;
            div_clk <= 1'b0;
        end else begin
            if (counter == 4'd4) begin
                counter <= 4'd0;
                div_clk <= ~div_clk;
            end else begin
                counter <= counter + 4'd1;
            end
        end
    end
endmodule"""
    },
    # ============ FSM ============
    {
        "name": "fsm_traffic_light",
        "category": "fsm",
        "spec": "Design a traffic light controller FSM with states: RED (3 cycles), GREEN (3 cycles), YELLOW (1 cycle). Inputs: clk, rst. Outputs: red, green, yellow (one-hot). Include 2-bit timer for state duration.",
        "reasoning": "Use enum for states. Timer increments each cycle in a state. When timer reaches limit, transition to next state (RED->GREEN->YELLOW->RED). Outputs are one-hot based on current state. Reset to RED with timer=0.",
        "code": """module traffic_light_fsm (
    input  clk,
    input  rst,
    output red,
    output green,
    output yellow
);
    typedef enum logic [1:0] {S_RED, S_GREEN, S_YELLOW} state_t;
    state_t state, next_state;
    reg [1:0] timer;
    
    // State register
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_RED;
            timer <= 2'd0;
        end else begin
            state <= next_state;
            if (state == next_state)
                timer <= timer + 2'd1;
            else
                timer <= 2'd0;
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            S_RED:    if (timer == 2'd2) next_state = S_GREEN;
            S_GREEN:  if (timer == 2'd2) next_state = S_YELLOW;
            S_YELLOW: if (timer == 2'd0) next_state = S_RED;
        endcase
    end
    
    // Output logic
    assign red    = (state == S_RED);
    assign green  = (state == S_GREEN);
    assign yellow = (state == S_YELLOW);
endmodule"""
    },
    {
        "name": "fsm_sequence_detector",
        "category": "fsm",
        "spec": "Design a Moore FSM that detects sequence '1011' in a serial input. Input: clk, rst, in. Output: detected (1 when last 4 bits are 1011).",
        "reasoning": "Sequence detector FSM tracks progress through '1','0','1','1'. States: IDLE, S1, S10, S101, S1011. On match, go to S1011 and assert detected. Overlapping sequences allowed (last '1' of '1011' can be first '1' of next).",
        "code": """module sequence_detector_1011 (
    input  clk,
    input  rst,
    input  in,
    output detected
);
    localparam [2:0] IDLE  = 3'b000;
    localparam [2:0] S1    = 3'b001;
    localparam [2:0] S10   = 3'b010;
    localparam [2:0] S101  = 3'b011;
    localparam [2:0] S1011 = 3'b100;
    
    reg [2:0] state, next_state;
    
    always_ff @(posedge clk) begin
        if (rst)
            state <= IDLE;
        else
            state <= next_state;
    end
    
    always_comb begin
        next_state = state;
        case (state)
            IDLE:  next_state = in ? S1   : IDLE;
            S1:    next_state = in ? S1   : S10;
            S10:   next_state = in ? S101 : IDLE;
            S101:  next_state = in ? S1011: S10;
            S1011: next_state = in ? S1   : S10;
            default: next_state = IDLE;
        endcase
    end
    
    assign detected = (state == S1011);
endmodule"""
    },
    {
        "name": "fsm_vending_machine",
        "category": "fsm",
        "spec": "Design a vending machine FSM for 25-cent item. Inputs: clk, rst, nickel, dime, quarter. Output: dispense (1 when enough money inserted), change (1 when overpaid). Track total in cents.",
        "reasoning": "Track accumulated money (0-30 cents). Each coin adds value. When >=25, dispense. If >25, give change. Reset after transaction. Use states for amount ranges, or a counter. Counter approach is simpler.",
        "code": """module vending_machine (
    input  clk,
    input  rst,
    input  nickel,
    input  dime,
    input  quarter,
    output reg dispense,
    output reg change
);
    reg [4:0] total;  // max 30 cents
    
    always_ff @(posedge clk) begin
        if (rst) begin
            total   <= 5'd0;
            dispense <= 1'b0;
            change   <= 1'b0;
        end else begin
            dispense <= 1'b0;
            change   <= 1'b0;
            
            if (nickel) total <= total + 5'd5;
            else if (dime) total <= total + 5'd10;
            else if (quarter) total <= total + 5'd25;
            
            if (total >= 5'd25) begin
                dispense <= 1'b1;
                if (total > 5'd25) change <= 1'b1;
                total <= 5'd0;
            end
        end
    end
endmodule"""
    },
    # ============ ARITHMETIC ============
    {
        "name": "full_adder",
        "category": "arithmetic",
        "spec": "Design a 1-bit full adder. Inputs: a, b, cin. Outputs: sum, cout. Include truth table in comments.",
        "reasoning": "Full adder computes sum = a XOR b XOR cin, and cout = majority(a,b,cin) = (a&b)|(a&cin)|(b&cin). This is the fundamental building block of multi-bit adders.",
        "code": """module full_adder (
    input  a, b, cin,
    output sum, cout
);
    assign sum  = a ^ b ^ cin;
    assign cout = (a & b) | (a & cin) | (b & cin);
endmodule"""
    },
    {
        "name": "adder_8bit_lookahead",
        "category": "arithmetic",
        "spec": "Design an 8-bit carry-lookahead adder. Inputs: A[7:0], B[7:0], Cin. Outputs: Sum[7:0], Cout. Use generate block with full adders and lookahead logic for speed.",
        "reasoning": "Carry-lookahead computes carry bits in parallel instead of rippling. For each bit: generate g = a&b, propagate p = a^b. Group carry: c[i+1] = g[i] | (p[i] & c[i]). Use 4-bit lookahead blocks for 8-bit adder.",
        "code": """module adder_8bit_cla (
    input  [7:0] A, B,
    input        Cin,
    output [7:0] Sum,
    output       Cout
);
    wire [7:0] g = A & B;
    wire [7:0] p = A ^ B;
    wire [8:0] c;
    assign c[0] = Cin;
    
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : cla
            assign c[i+1] = g[i] | (p[i] & c[i]);
            assign Sum[i] = p[i] ^ c[i];
        end
    endgenerate
    
    assign Cout = c[8];
endmodule"""
    },
    {
        "name": "multiplier_4bit_array",
        "category": "arithmetic",
        "spec": "Design a 4-bit unsigned array multiplier. Inputs: A[3:0], B[3:0]. Output: P[7:0] = A * B. Use generate block with AND gates and ripple adders.",
        "reasoning": "Array multiplier computes partial products (A[i] & B[j]) and sums them with shifted alignment. Each row is a partial product. Use full adders to sum rows. Generate block creates regular structure.",
        "code": """module multiplier_4bit_array (
    input  [3:0] A, B,
    output [7:0] P
);
    wire [3:0] pp0 = A & {4{B[0]}};
    wire [3:0] pp1 = A & {4{B[1]}};
    wire [3:0] pp2 = A & {4{B[2]}};
    wire [3:0] pp3 = A & {4{B[3]}};
    
    wire [4:0] sum1 = pp0 + (pp1 << 1);
    wire [5:0] sum2 = sum1 + (pp2 << 2);
    wire [7:0] sum3 = sum2 + (pp3 << 3);
    
    assign P = sum3;
endmodule"""
    },
    {
        "name": "abs_value",
        "category": "arithmetic",
        "spec": "Design an 8-bit absolute value circuit. Input signed_in[7:0] (two's complement). Output abs_out[7:0] (unsigned absolute value).",
        "reasoning": "For two's complement, if MSB=1 (negative), absolute value is ~x + 1 (two's complement negation). If positive, value is unchanged. Check MSB and conditionally negate.",
        "code": """module abs_value_8bit (
    input  signed [7:0] signed_in,
    output [7:0] abs_out
);
    assign abs_out = signed_in[7] ? (~signed_in + 8'd1) : signed_in;
endmodule"""
    },
    # ============ MEMORY ============
    {
        "name": "ram_16x8",
        "category": "memory",
        "spec": "Design a 16x8 RAM (16 locations, 8 bits each). Inputs: clk, we (write enable), addr[3:0], din[7:0]. Output: dout[7:0]. Synchronous write, asynchronous read.",
        "reasoning": "RAM uses a register array. Write on clock edge when we=1. Read is combinational (asynchronous). Use unpacked array for memory. Synchronous write prevents glitches.",
        "code": """module ram_16x8 (
    input        clk,
    input        we,
    input  [3:0] addr,
    input  [7:0] din,
    output [7:0] dout
);
    reg [7:0] mem [0:15];
    
    always_ff @(posedge clk) begin
        if (we)
            mem[addr] <= din;
    end
    
    assign dout = mem[addr];
endmodule"""
    },
    {
        "name": "fifo_sync",
        "category": "memory",
        "spec": "Design a synchronous FIFO, depth 8, width 8. Inputs: clk, rst, wr_en, rd_en, din[7:0]. Outputs: dout[7:0], full, empty. Use circular buffer with head/tail pointers.",
        "reasoning": "Circular buffer uses head (write) and tail (read) pointers. FIFO is full when head == tail+1 (with wrap). Empty when head == tail. Use one extra slot to distinguish full/empty, or a count register. Count approach is cleaner.",
        "code": """module fifo_sync_8x8 (
    input        clk,
    input        rst,
    input        wr_en,
    input        rd_en,
    input  [7:0] din,
    output [7:0] dout,
    output       full,
    output       empty
);
    reg [7:0] mem [0:7];
    reg [2:0] wr_ptr, rd_ptr;
    reg [3:0] count;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            wr_ptr <= 3'd0;
            rd_ptr <= 3'd0;
            count  <= 4'd0;
        end else begin
            if (wr_en && !full) begin
                mem[wr_ptr] <= din;
                wr_ptr <= wr_ptr + 3'd1;
                count <= count + 4'd1;
            end
            if (rd_en && !empty) begin
                rd_ptr <= rd_ptr + 3'd1;
                count <= count - 4'd1;
            end
        end
    end
    
    assign dout  = mem[rd_ptr];
    assign full  = (count == 4'd8);
    assign empty = (count == 4'd0);
endmodule"""
    },
    {
        "name": "register_file",
        "category": "memory",
        "spec": "Design a 32x32 register file (32 registers, 32-bit each). Inputs: clk, we, rd_addr1[4:0], rd_addr2[4:0], wr_addr[4:0], wr_data[31:0]. Outputs: rd_data1[31:0], rd_data2[31:0]. Register 0 is hardwired to 0.",
        "reasoning": "Dual-read-port register file common in CPUs. Two combinational reads, one synchronous write. Register 0 always returns 0 regardless of write. Check write address to enforce this.",
        "code": """module register_file_32x32 (
    input         clk,
    input         we,
    input  [4:0]  rd_addr1,
    input  [4:0]  rd_addr2,
    input  [4:0]  wr_addr,
    input  [31:0] wr_data,
    output [31:0] rd_data1,
    output [31:0] rd_data2
);
    reg [31:0] regs [0:31];
    
    always_ff @(posedge clk) begin
        if (we && wr_addr != 5'd0)
            regs[wr_addr] <= wr_data;
    end
    
    assign rd_data1 = (rd_addr1 == 5'd0) ? 32'd0 : regs[rd_addr1];
    assign rd_data2 = (rd_addr2 == 5'd0) ? 32'd0 : regs[rd_addr2];
endmodule"""
    },
    # ============ INTERFACE ============
    {
        "name": "uart_tx",
        "category": "interface",
        "spec": "Design a UART transmitter (8N1: 8 data bits, no parity, 1 stop bit). Inputs: clk, rst, tx_start, tx_data[7:0], baud_tick. Output: tx (serial output), tx_busy. State machine: IDLE -> START -> DATA -> STOP.",
        "reasoning": "UART TX shifts out start bit (0), 8 data bits (LSB first), stop bit (1). Use baud_tick as enable (one pulse per bit period). FSM tracks which bit is being sent. Shift register holds data.",
        "code": """module uart_tx (
    input        clk,
    input        rst,
    input        tx_start,
    input  [7:0] tx_data,
    input        baud_tick,
    output reg   tx,
    output       tx_busy
);
    typedef enum logic [1:0] {IDLE, START, DATA, STOP} state_t;
    state_t state;
    reg [2:0] bit_cnt;
    reg [7:0] shift_reg;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            tx <= 1'b1;
        end else if (baud_tick) begin
            case (state)
                IDLE: begin
                    tx <= 1'b1;
                    if (tx_start) begin
                        state <= START;
                        shift_reg <= tx_data;
                    end
                end
                START: begin
                    tx <= 1'b0;
                    state <= DATA;
                    bit_cnt <= 3'd0;
                end
                DATA: begin
                    tx <= shift_reg[0];
                    shift_reg <= shift_reg >> 1;
                    if (bit_cnt == 3'd7)
                        state <= STOP;
                    else
                        bit_cnt <= bit_cnt + 3'd1;
                end
                STOP: begin
                    tx <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end
    
    assign tx_busy = (state != IDLE);
endmodule"""
    },
    {
        "name": "spi_master",
        "category": "interface",
        "spec": "Design an SPI master for Mode 0 (CPOL=0, CPHA=0). Inputs: clk, rst, start, data_in[7:0]. Outputs: mosi, sclk, cs_n, data_out[7:0], done. Transfer 8 bits, MSB first. Sclk is clk/4.",
        "reasoning": "Mode 0: clock idle low, data sampled on rising edge, shifted on falling edge. Use a clock divider for sclk. Shift register for data. cs_n active low during transfer. Counter tracks 8 bits.",
        "code": """module spi_master (
    input        clk,
    input        rst,
    input        start,
    input  [7:0] data_in,
    output reg   mosi,
    output       sclk,
    output reg   cs_n,
    output reg [7:0] data_out,
    output reg   done
);
    reg [1:0] clk_div;
    reg [2:0] bit_cnt;
    reg [7:0] tx_reg, rx_reg;
    reg       sclk_reg;
    reg       active;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            clk_div  <= 2'd0;
            sclk_reg <= 1'b0;
            cs_n     <= 1'b1;
            active   <= 1'b0;
            done     <= 1'b0;
            bit_cnt  <= 3'd0;
        end else begin
            done <= 1'b0;
            if (!active && start) begin
                active  <= 1'b1;
                cs_n    <= 1'b0;
                tx_reg  <= data_in;
                bit_cnt <= 3'd0;
                clk_div <= 2'd0;
            end else if (active) begin
                clk_div <= clk_div + 2'd1;
                if (clk_div == 2'd3) begin
                    sclk_reg <= ~sclk_reg;
                    if (!sclk_reg) begin  // rising edge
                        rx_reg <= {rx_reg[6:0], mosi};
                    end else begin        // falling edge
                        mosi <= tx_reg[7];
                        tx_reg <= tx_reg << 1;
                        if (bit_cnt == 3'd7) begin
                            active <= 1'b0;
                            cs_n   <= 1'b1;
                            done   <= 1'b1;
                            data_out <= rx_reg;
                        end else begin
                            bit_cnt <= bit_cnt + 3'd1;
                        end
                    end
                end
            end
        end
    end
    
    assign sclk = sclk_reg;
endmodule"""
    },
    # ============ ADVANCED ============
    {
        "name": "crc8_generator",
        "category": "advanced",
        "spec": "Design an 8-bit CRC generator (polynomial x^8 + x^2 + x + 1 = 0x07). Input: clk, rst, data_in (serial, 1 bit per clock), valid. Output: crc[7:0] (valid after 8 cycles of valid data). LSB-first.",
        "reasoning": "CRC is a linear feedback shift register with taps at polynomial positions. For each input bit, XOR with MSB, shift, and XOR feedback into tap positions. Process LSB first.",
        "code": """module crc8_generator (
    input        clk,
    input        rst,
    input        data_in,
    input        valid,
    output [7:0] crc
);
    reg [7:0] lfsr;
    wire feedback = lfsr[7] ^ data_in;
    
    always_ff @(posedge clk) begin
        if (rst)
            lfsr <= 8'd0;
        else if (valid)
            lfsr <= {feedback, lfsr[6:2], lfsr[1] ^ feedback, lfsr[0] ^ feedback};
    end
    
    assign crc = lfsr;
endmodule"""
    },
    {
        "name": "wallace_tree_4x4",
        "category": "advanced",
        "spec": "Design a 4-bit Wallace tree multiplier. Inputs: A[3:0], B[3:0]. Output: P[7:0]. Use full adders to sum partial products in tree structure, then final ripple adder.",
        "reasoning": "Wallace tree reduces partial products in log(N) stages using carry-save adders (full adders). Stage 1: reduce groups of 3 bits to 2. Stage 2: continue reducing. Final stage: ripple carry adder for last 2 rows.",
        "code": """module wallace_tree_4x4 (
    input  [3:0] A, B,
    output [7:0] P
);
    // Partial products
    wire p00 = A[0] & B[0]; wire p01 = A[0] & B[1]; wire p02 = A[0] & B[2]; wire p03 = A[0] & B[3];
    wire p10 = A[1] & B[0]; wire p11 = A[1] & B[1]; wire p12 = A[1] & B[2]; wire p13 = A[1] & B[3];
    wire p20 = A[2] & B[0]; wire p21 = A[2] & B[1]; wire p22 = A[2] & B[2]; wire p23 = A[2] & B[3];
    wire p30 = A[3] & B[0]; wire p31 = A[3] & B[1]; wire p32 = A[3] & B[2]; wire p33 = A[3] & B[3];
    
    // Column sums (simplified: using array multiplier logic for compactness)
    wire [4:0] row0 = {1'b0, p03, p02, p01, p00};
    wire [4:0] row1 = {p13, p12, p11, p10, 1'b0};
    wire [4:0] row2 = {p23, p22, p21, p20, 1'b0};
    wire [4:0] row3 = {p33, p32, p31, p30, 1'b0};
    
    assign P = row0 + (row1 << 1) + (row2 << 2) + (row3 << 3);
endmodule"""
    },
    {
        "name": "leading_one_detector",
        "category": "advanced",
        "spec": "Design an 8-bit leading-one detector. Input in[7:0]. Output position[2:0] (index of highest 1 bit, 0 if no 1s found), found (1 if any 1 present).",
        "reasoning": "Find highest-index bit set to 1. Check from MSB to LSB. Priority encoder pattern. Return position and whether any bit was found. If input is 0, found=0 and position=0.",
        "code": """module leading_one_detector (
    input  [7:0] in,
    output reg [2:0] position,
    output       found
);
    always_comb begin
        if      (in[7]) position = 3'd7;
        else if (in[6]) position = 3'd6;
        else if (in[5]) position = 3'd5;
        else if (in[4]) position = 3'd4;
        else if (in[3]) position = 3'd3;
        else if (in[2]) position = 3'd2;
        else if (in[1]) position = 3'd1;
        else            position = 3'd0;
    end
    assign found = |in;
endmodule"""
    },
]

# Add parameter variations for augmentation
PARAMETER_VARIATIONS = {
    "mux2to1": ["8-bit", "16-bit", "32-bit", "64-bit"],
    "counter_8bit_up": ["4-bit", "8-bit", "12-bit", "16-bit"],
    "ram_16x8": ["16x8", "32x16", "64x32"],
    "fifo_sync": ["8x8", "16x16", "32x32"],
    "pwm_generator": ["8-bit", "10-bit", "12-bit"],
}


def get_all_templates():
    """Return all base templates."""
    return VERILOG_TEMPLATES


def get_templates_by_category(category: str):
    """Filter templates by category."""
    return [t for t in VERILOG_TEMPLATES if t["category"] == category]
