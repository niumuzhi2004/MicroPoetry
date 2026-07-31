// look-up table used in the softmax actuator
// used for determining the reciprocal of total
// i.e., 1 / total in softmax()

import proj_pkg::*;

module recip_lut (
    input  logic [5:0]  addr,
    output logic [15:0] data
);

    always_comb begin
        case (addr)
            6'd0:  data = 16'd32768;
            6'd1:  data = 16'd31775;
            6'd2:  data = 16'd30840;
            6'd3:  data = 16'd29959;
            6'd4:  data = 16'd29127;
            6'd5:  data = 16'd28340;
            6'd6:  data = 16'd27594;
            6'd7:  data = 16'd26887;
            6'd8:  data = 16'd26214;
            6'd9:  data = 16'd25575;
            6'd10: data = 16'd24966;
            6'd11: data = 16'd24385;
            6'd12: data = 16'd23831;
            6'd13: data = 16'd23302;
            6'd14: data = 16'd22795;
            6'd15: data = 16'd22310;
            6'd16: data = 16'd21845;
            6'd17: data = 16'd21400;
            6'd18: data = 16'd20972;
            6'd19: data = 16'd20560;
            6'd20: data = 16'd20165;
            6'd21: data = 16'd19784;
            6'd22: data = 16'd19418;
            6'd23: data = 16'd19065;
            6'd24: data = 16'd18725;
            6'd25: data = 16'd18396;
            6'd26: data = 16'd18079;
            6'd27: data = 16'd17772;
            6'd28: data = 16'd17476;
            6'd29: data = 16'd17190;
            6'd30: data = 16'd16913;
            6'd31: data = 16'd16644;
            6'd32: data = 16'd16384;
            default: data = 16'd0;
        endcase
    end
    
endmodule