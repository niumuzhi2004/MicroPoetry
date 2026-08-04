// look-up table used in the softmax actuator
// used for determining the exponential of 
// the difference between logits and the maximum logit
// i.e., (val - max_val).exp() in softmax()

import proj_pkg::*;

module exp_lut(
    input  logic [5:0]  addr,
    output logic [15:0] data
);

    always_comb begin
        case (addr)
            6'd0:  data = 16'd11;
            6'd1:  data = 16'd14;
            6'd2:  data = 16'd18;
            6'd3:  data = 16'd23;
            6'd4:  data = 16'd30;
            6'd5:  data = 16'd38;
            6'd6:  data = 16'd49;
            6'd7:  data = 16'd63;
            6'd8:  data = 16'd81;
            6'd9:  data = 16'd104;
            6'd10: data = 16'd134;
            6'd11: data = 16'd172;
            6'd12: data = 16'd221;
            6'd13: data = 16'd283;
            6'd14: data = 16'd364;
            6'd15: data = 16'd467;
            6'd16: data = 16'd600;
            6'd17: data = 16'd771;
            6'd18: data = 16'd990;
            6'd19: data = 16'd1271;
            6'd20: data = 16'd1631;
            6'd21: data = 16'd2095;
            6'd22: data = 16'd2690;
            6'd23: data = 16'd3454;
            6'd24: data = 16'd4435;
            6'd25: data = 16'd5694;
            6'd26: data = 16'd7312;
            6'd27: data = 16'd9388;
            6'd28: data = 16'd12055;
            6'd29: data = 16'd15479;
            6'd30: data = 16'd19875;
            6'd31: data = 16'd25520;
            6'd32: data = 16'd32768;
            default: data = 16'd0;
        endcase
    end
    
endmodule