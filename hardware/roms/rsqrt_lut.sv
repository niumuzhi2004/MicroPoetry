import proj_pkg::*;

module rsqrt_lut (
    input  norm_param_t param,
    input  logic [4:0]  addr,
    output logic [15:0] data
);

    always_comb begin
        case (param)
            EMB_NORM: begin
                case (addr)
                    5'd0:  data = 16'd57759;
                    5'd1:  data = 16'd33347;
                    5'd2:  data = 16'd25831;
                    5'd3:  data = 16'd21831;
                    5'd4:  data = 16'd19253;
                    5'd5:  data = 16'd17415;
                    5'd6:  data = 16'd16020;
                    5'd7:  data = 16'd14913;
                    5'd8:  data = 16'd14009;
                    5'd9:  data = 16'd13251;
                    5'd10: data = 16'd12604;
                    5'd11: data = 16'd12044;
                    5'd12: data = 16'd11552;
                    5'd13: data = 16'd11116;
                    5'd14: data = 16'd10726;
                    5'd15: data = 16'd10374;
                    5'd16: data = 16'd10055;
                    5'd17: data = 16'd9763;
                    5'd18: data = 16'd9496;
                    5'd19: data = 16'd9249;
                    5'd20: data = 16'd9020;
                    5'd21: data = 16'd8808;
                    5'd22: data = 16'd8610;
                    5'd23: data = 16'd8425;
                    5'd24: data = 16'd8251;
                    5'd25: data = 16'd8088;
                    5'd26: data = 16'd7934;
                    5'd27: data = 16'd7788;
                    5'd28: data = 16'd7650;
                    5'd29: data = 16'd7520;
                    5'd30: data = 16'd7395;
                    5'd31: data = 16'd7277;
                    default: data = 16'd0;
                endcase
            end

            X_NORM: begin
                case (addr)
                    5'd0:  data = 16'd13380;
                    5'd1:  data = 16'd7725;
                    5'd2:  data = 16'd5984;
                    5'd3:  data = 16'd5057;
                    5'd4:  data = 16'd4460;
                    5'd5:  data = 16'd4034;
                    5'd6:  data = 16'd3711;
                    5'd7:  data = 16'd3455;
                    5'd8:  data = 16'd3245;
                    5'd9:  data = 16'd3070;
                    5'd10: data = 16'd2920;
                    5'd11: data = 16'd2790;
                    5'd12: data = 16'd2676;
                    5'd13: data = 16'd2575;
                    5'd14: data = 16'd2485;
                    5'd15: data = 16'd2403;
                    5'd16: data = 16'd2329;
                    5'd17: data = 16'd2262;
                    5'd18: data = 16'd2200;
                    5'd19: data = 16'd2143;
                    5'd20: data = 16'd2090;
                    5'd21: data = 16'd2040;
                    5'd22: data = 16'd1995;
                    5'd23: data = 16'd1952;
                    5'd24: data = 16'd1911;
                    5'd25: data = 16'd1874;
                    5'd26: data = 16'd1838;
                    5'd27: data = 16'd1804;
                    5'd28: data = 16'd1772;
                    5'd29: data = 16'd1742;
                    5'd30: data = 16'd1713;
                    5'd31: data = 16'd1686;
                    default: data = 16'd0;
                endcase
            end

            PRE_MLP_NORM: begin
                case (addr)
                    5'd0:  data = 16'd13755;
                    5'd1:  data = 16'd7941;
                    5'd2:  data = 16'd6151;
                    5'd3:  data = 16'd5199;
                    5'd4:  data = 16'd4585;
                    5'd5:  data = 16'd4147;
                    5'd6:  data = 16'd3815;
                    5'd7:  data = 16'd3551;
                    5'd8:  data = 16'd3336;
                    5'd9:  data = 16'd3156;
                    5'd10: data = 16'd3002;
                    5'd11: data = 16'd2868;
                    5'd12: data = 16'd2751;
                    5'd13: data = 16'd2647;
                    5'd14: data = 16'd2554;
                    5'd15: data = 16'd2470;
                    5'd16: data = 16'd2394;
                    5'd17: data = 16'd2325;
                    5'd18: data = 16'd2261;
                    5'd19: data = 16'd2203;
                    5'd20: data = 16'd2148;
                    5'd21: data = 16'd2098;
                    5'd22: data = 16'd2050;
                    5'd23: data = 16'd2006;
                    5'd24: data = 16'd1965;
                    5'd25: data = 16'd1926;
                    5'd26: data = 16'd1889;
                    5'd27: data = 16'd1855;
                    5'd28: data = 16'd1822;
                    5'd29: data = 16'd1791;
                    5'd30: data = 16'd1761;
                    5'd31: data = 16'd1733;
                    default: data = 16'd0;
                endcase
            end

            FINAL_NORM: begin
                case (addr)
                    5'd0:  data = 16'd9046;
                    5'd1:  data = 16'd5223;
                    5'd2:  data = 16'd4046;
                    5'd3:  data = 16'd3419;
                    5'd4:  data = 16'd3015;
                    5'd5:  data = 16'd2728;
                    5'd6:  data = 16'd2509;
                    5'd7:  data = 16'd2336;
                    5'd8:  data = 16'd2194;
                    5'd9:  data = 16'd2075;
                    5'd10: data = 16'd1974;
                    5'd11: data = 16'd1886;
                    5'd12: data = 16'd1809;
                    5'd13: data = 16'd1741;
                    5'd14: data = 16'd1680;
                    5'd15: data = 16'd1625;
                    5'd16: data = 16'd1575;
                    5'd17: data = 16'd1529;
                    5'd18: data = 16'd1487;
                    5'd19: data = 16'd1449;
                    5'd20: data = 16'd1413;
                    5'd21: data = 16'd1380;
                    5'd22: data = 16'd1349;
                    5'd23: data = 16'd1320;
                    5'd24: data = 16'd1292;
                    5'd25: data = 16'd1267;
                    5'd26: data = 16'd1243;
                    5'd27: data = 16'd1220;
                    5'd28: data = 16'd1198;
                    5'd29: data = 16'd1178;
                    5'd30: data = 16'd1158;
                    5'd31: data = 16'd1140;
                    default: data = 16'd0;
                endcase
            end
            
            default: data = 16'd0;
        endcase
    end
    
endmodule