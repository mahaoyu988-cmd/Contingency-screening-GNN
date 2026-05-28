define_constants;
mpopt_opf = mpoption('verbose', 0, 'out.all', 0);
mpopt_pf  = mpoption('verbose', 0, 'out.all', 0);

%% ============================================================
% Base case preparation
%% ============================================================
mpc0 = loadcase('case118');

PD0 = mpc0.bus(:, PD);
QD0 = mpc0.bus(:, QD);

%% -----------------------------
% Assign branch thermal ratings using MAX terminal voltage
%% -----------------------------
base_kv = mpc0.bus(:, BASE_KV);
nbr = size(mpc0.branch, 1);
Smax = zeros(nbr, 1);

for l = 1:nbr
    fb = mpc0.branch(l, F_BUS);
    tb = mpc0.branch(l, T_BUS);

    kv_f = base_kv(fb);
    kv_t = base_kv(tb);
    kv   = max(kv_f, kv_t);

    if abs(kv - 345) < 1e-3
        Smax(l) = 700;
    elseif abs(kv - 161) < 1e-3
        Smax(l) = 220;
    else
        Smax(l) = 180;
    end
end

mpc0.branch(:, RATE_A) = Smax;

%% -----------------------------
% Renewable replacement settings
%% -----------------------------
wind_bus  = 10;
solar_bus = 26;

pf_allow = 0.9;
tanphi   = tan(acos(pf_allow));

Vset_wind  = 1.00;
Vset_solar = 1.00;

g10 = find(mpc0.gen(:, GEN_BUS) == wind_bus  & mpc0.gen(:, GEN_STATUS) > 0);
g26 = find(mpc0.gen(:, GEN_BUS) == solar_bus & mpc0.gen(:, GEN_STATUS) > 0);

if isempty(g10)
    error('No in-service generator found at bus %d.', wind_bus);
end
if isempty(g26)
    error('No in-service generator found at bus %d.', solar_bus);
end

keep10 = g10(1);
keep26 = g26(1);

Pmax10_old = sum(mpc0.gen(g10, PMAX));
Pmax26_old = sum(mpc0.gen(g26, PMAX));

Pinst_wind  = Pmax10_old;
Pinst_solar = Pmax26_old;

Qcap_wind  = Pinst_wind  * tanphi;
Qcap_solar = Pinst_solar * tanphi;

if length(g10) > 1
    mpc0.gen(g10(2:end), GEN_STATUS) = 0;
    mpc0.gen(g10(2:end), PG)         = 0;
    mpc0.gen(g10(2:end), QG)         = 0;
end

if length(g26) > 1
    mpc0.gen(g26(2:end), GEN_STATUS) = 0;
    mpc0.gen(g26(2:end), PG)         = 0;
    mpc0.gen(g26(2:end), QG)         = 0;
end

mpc0.bus(wind_bus,  BUS_TYPE) = PV;
mpc0.bus(solar_bus, BUS_TYPE) = PV;

if isfield(mpc0, 'gencost') && size(mpc0.gencost,1) >= size(mpc0.gen,1)
    for g = [keep10, keep26]
        mpc0.gencost(g, MODEL)    = 2;
        mpc0.gencost(g, STARTUP)  = 0;
        mpc0.gencost(g, SHUTDOWN) = 0;
        mpc0.gencost(g, NCOST)    = 3;
        mpc0.gencost(g, COST)     = 0;
        mpc0.gencost(g, COST+1)   = 0.01;
        mpc0.gencost(g, COST+2)   = 0;
    end
end

%% ============================================================
% Seasonal max/min curves
%% ============================================================
% ---------- Load ----------
load_summer = [ ...
    0.72 0.69 0.67 0.66 0.66 0.69 0.75 0.81 0.86 0.90 0.93 0.96 ...
    0.98 1.00 1.00 0.99 0.97 0.94 0.90 0.85 0.79 0.73 0.69 0.66 ]' * 1.75;

load_winter = [ ...
    0.91 0.89 0.87 0.86 0.86 0.88 0.93 0.98 1.00 0.99 0.98 0.97 ...
    0.96 0.95 0.94 0.94 0.95 0.97 0.99 1.00 0.98 0.95 0.93 0.90 ]' * 1.4;

load_spring = [ ...
    0.85 0.83 0.81 0.80 0.80 0.83 0.89 0.94 0.96 0.97 0.98 0.98 ...
    0.99 0.99 1.00 1.00 1.00 1.00 0.99 0.97 0.94 0.90 0.87 0.84 ]';

load_fall = [ ...
    0.86 0.84 0.82 0.81 0.81 0.84 0.89 0.94 0.96 0.97 0.98 0.98 ...
    0.99 0.99 1.00 1.00 1.00 0.99 0.97 0.94 0.90 0.86 0.82 0.79 ]';

LoadMat = [load_summer, load_winter, load_spring, load_fall];
Lmin = min(LoadMat, [], 2);
Lmax = max(LoadMat, [], 2);

% ---------- PV ----------
pv_summer = [ ...
    0 0 0 0 0 0.02 0.08 0.18 0.35 0.55 0.72 0.85 ...
    0.95 1.00 0.96 0.86 0.68 0.42 0.18 0.04 0 0 0 0 ]';

pv_winter = [ ...
    0 0 0 0 0 0 0 0.04 0.12 0.24 0.38 0.50 ...
    0.58 0.55 0.44 0.28 0.10 0.01 0 0 0 0 0 0 ]';

pv_spring = [ ...
    0 0 0 0 0 0.01 0.05 0.14 0.28 0.46 0.64 0.79 ...
    0.88 0.90 0.82 0.66 0.44 0.20 0.05 0 0 0 0 0 ]';

pv_fall = [ ...
    0 0 0 0 0 0.00 0.03 0.10 0.22 0.38 0.54 0.68 ...
    0.78 0.76 0.66 0.50 0.28 0.10 0.01 0 0 0 0 0 ]';

PVMat = [pv_summer, pv_winter, pv_spring, pv_fall];
PVmin = min(PVMat, [], 2);
PVmax = max(PVMat, [], 2);

% ---------- Wind ----------
wind_summer = [ ...
    0.55 0.57 0.59 0.60 0.61 0.60 0.57 0.53 0.48 0.44 0.40 0.37 ...
    0.35 0.35 0.36 0.39 0.43 0.47 0.51 0.54 0.56 0.57 0.56 0.55 ]';

wind_winter = [ ...
    0.82 0.84 0.86 0.88 0.89 0.87 0.83 0.77 0.71 0.65 0.61 0.58 ...
    0.57 0.58 0.60 0.64 0.69 0.74 0.78 0.81 0.83 0.84 0.83 0.82 ]';

wind_spring = [ ...
    0.72 0.75 0.78 0.80 0.81 0.79 0.74 0.68 0.62 0.57 0.53 0.50 ...
    0.49 0.50 0.52 0.56 0.61 0.66 0.70 0.72 0.74 0.75 0.74 0.73 ]';

wind_fall = [ ...
    0.66 0.69 0.72 0.74 0.75 0.73 0.69 0.64 0.58 0.53 0.49 0.46 ...
    0.45 0.46 0.48 0.52 0.57 0.61 0.65 0.67 0.69 0.70 0.69 0.67 ]';

WindMat = [wind_summer, wind_winter, wind_spring, wind_fall];
Wmin = min(WindMat, [], 2);
Wmax = max(WindMat, [], 2);

%% ============================================================
% Original 8 scenarios
%% ============================================================
scenario_defs = { ...
    'Lmin_Wmin_Smin', Lmin, Wmin, PVmin; ...
    'Lmin_Wmin_Smax', Lmin, Wmin, PVmax; ...
    'Lmin_Wmax_Smin', Lmin, Wmax, PVmin; ...
    'Lmin_Wmax_Smax', Lmin, Wmax, PVmax; ...
    'Lmax_Wmin_Smin', Lmax, Wmin, PVmin; ...
    'Lmax_Wmin_Smax', Lmax, Wmin, PVmax; ...
    'Lmax_Wmax_Smin', Lmax, Wmax, PVmin; ...
    'Lmax_Wmax_Smax', Lmax, Wmax, PVmax};

nScen = size(scenario_defs, 1);

%% ============================================================
% Contingencies to study
%% ============================================================
cont_list = [74 25 60 126 142 38 93 97];
nCont = numel(cont_list);

%% ============================================================
% Static export tables
%% ============================================================
case118_branch_data = table( ...
    mpc0.branch(:, F_BUS), ...
    mpc0.branch(:, T_BUS), ...
    mpc0.branch(:, BR_R), ...
    mpc0.branch(:, BR_X), ...
    mpc0.branch(:, BR_B), ...
    mpc0.branch(:, RATE_A), ...
    mpc0.branch(:, TAP), ...
    'VariableNames', {'from_bus','to_bus','r','x','b','rateA','ratio'});

nb = size(mpc0.bus,1);
ng = size(mpc0.gen,1);

is_slack = double(mpc0.bus(:, BUS_TYPE) == REF);
is_pv    = double(mpc0.bus(:, BUS_TYPE) == PV);
is_pq    = double(mpc0.bus(:, BUS_TYPE) == PQ);

case118_bus_type = table( ...
    mpc0.bus(:, BUS_I), is_slack, is_pv, is_pq, ...
    'VariableNames', {'bus','is_slack','is_pv','is_pq'});

%% ============================================================
% Build new contingency PF dataset
%% ============================================================
train_rows = {};
dispatch_rows = {};
meta_rows = {};
violation_rows = {};

sample_id = 0;

for s = 1:nScen
    scen_name     = scenario_defs{s,1};
    load_curve_s  = scenario_defs{s,2};
    wind_curve_s  = scenario_defs{s,3};
    solar_curve_s = scenario_defs{s,4};

    fprintf('\n================ Scenario %d: %s ================\n', s, scen_name);

    for h = 1:24
        %% ----------------------------------------------------
        % Step 1: Build OPF case and run ACOPF
        %% ----------------------------------------------------
        mpc_opf = mpc0;

        alpha = load_curve_s(h);
        mpc_opf.bus(:, PD) = alpha * PD0;
        mpc_opf.bus(:, QD) = alpha * QD0;

        Pavail_wind  = max(0, wind_curve_s(h))  * Pinst_wind;
        Pavail_solar = max(0, solar_curve_s(h)) * Pinst_solar;

        % Wind at bus 10
        mpc_opf.gen(keep10, PG)         = Pavail_wind;
        mpc_opf.gen(keep10, QG)         = 0;
        mpc_opf.gen(keep10, QMAX)       =  Qcap_wind;
        mpc_opf.gen(keep10, QMIN)       = -Qcap_wind;
        mpc_opf.gen(keep10, VG)         = Vset_wind;
        mpc_opf.gen(keep10, PMAX)       = Pavail_wind;
        mpc_opf.gen(keep10, PMIN)       = Pavail_wind;   % fixed at upper dispatch
        mpc_opf.gen(keep10, GEN_STATUS) = 1;

        % Solar at bus 26
        mpc_opf.gen(keep26, PG)         = Pavail_solar;
        mpc_opf.gen(keep26, QG)         = 0;
        mpc_opf.gen(keep26, QMAX)       =  Qcap_solar;
        mpc_opf.gen(keep26, QMIN)       = -Qcap_solar;
        mpc_opf.gen(keep26, VG)         = Vset_solar;
        mpc_opf.gen(keep26, PMAX)       = Pavail_solar;
        mpc_opf.gen(keep26, PMIN)       = Pavail_solar;
        mpc_opf.gen(keep26, GEN_STATUS) = 1;

        r_opf = runopf(mpc_opf, mpopt_opf);

        if ~r_opf.success
            fprintf('Scenario %s hour %2d: ACOPF failed.\n', scen_name, h);
            continue;
        end

        % ------------------------------------------------
        % Identify slack generator for post-contingency PF
        % ------------------------------------------------
        slack_bus = r_opf.bus(r_opf.bus(:, BUS_TYPE) == REF, BUS_I);
        
        if isempty(slack_bus)
            error('No slack/reference bus found for scenario %s hour %d.', scen_name, h);
        end
        
        slack_bus = slack_bus(1);
        
        slack_gen_idx = find(r_opf.gen(:, GEN_BUS) == slack_bus & r_opf.gen(:, GEN_STATUS) > 0);
        
        if isempty(slack_gen_idx)
            error('No in-service generator found at slack bus %d for scenario %s hour %d.', ...
                slack_bus, scen_name, h);
        end
        
        slack_gen_idx = slack_gen_idx(1);
        
        online_gen_idx = find(r_opf.gen(:, GEN_STATUS) > 0);
        fixed_gen_idx  = setdiff(online_gen_idx, slack_gen_idx);

        %% ----------------------------------------------------
        % Step 2: For each outage, run ACPF with:
        %   - PG fixed to ACOPF PG
        %   - QG = 0 initial
        %   - VG = 1.0 initial
        %   - VM and VA initial guess from ACOPF solution
        %% ----------------------------------------------------
        for c = 1:nCont
            cont_idx = cont_list(c);

            mpc_pf = mpc_opf;

            % Apply ACOPF solved state as base operating point
            mpc_pf.bus(:, VM) = r_opf.bus(:, VM);
            mpc_pf.bus(:, VA) = r_opf.bus(:, VA);
            
            % Keep ACOPF active dispatch as starting point
            mpc_pf.gen(:, PG) = r_opf.gen(:, PG);
            
            % Set generator initialization as requested
            mpc_pf.gen(:, QG) = 0;
            mpc_pf.gen(:, VG) = 1.0;

            % Fix only non-slack generators
            mpc_pf.gen(fixed_gen_idx, PMAX) = r_opf.gen(fixed_gen_idx, PG);
            mpc_pf.gen(fixed_gen_idx, PMIN) = r_opf.gen(fixed_gen_idx, PG);
                        
            % ------------------------------------------------
            % Do NOT fix slack generator dispatch
            % Fix only non-slack online generators at ACOPF PG
            % ------------------------------------------------
            slack_bus = r_opf.bus(r_opf.bus(:, BUS_TYPE) == REF, BUS_I);
            
            if isempty(slack_bus)
                error('No slack/reference bus found for scenario %s hour %d.', scen_name, h);
            end
            
            slack_bus = slack_bus(1);
            
            slack_gen_idx = find(mpc_pf.gen(:, GEN_BUS) == slack_bus & mpc_pf.gen(:, GEN_STATUS) > 0);
            
            if isempty(slack_gen_idx)
                error('No in-service generator found at slack bus %d for scenario %s hour %d.', ...
                    slack_bus, scen_name, h);
            end
            
            slack_gen_idx = slack_gen_idx(1);
            
            online_gen_idx = find(mpc_pf.gen(:, GEN_STATUS) > 0);
            fixed_gen_idx  = setdiff(online_gen_idx, slack_gen_idx);
            
            % Fix only non-slack generators
            mpc_pf.gen(fixed_gen_idx, PMAX) = r_opf.gen(fixed_gen_idx, PG);
            mpc_pf.gen(fixed_gen_idx, PMIN) = r_opf.gen(fixed_gen_idx, PG);
            
            % Leave slack generator PMAX/PMIN unchanged
            % Branch outage
            mpc_pf.branch(cont_idx, BR_STATUS) = 0;

                        % Run ACPF
            r_pf = runpf(mpc_pf, mpopt_pf);

            sample_id = sample_id + 1;

            if ~r_pf.success
                fprintf('Scenario %s hour %2d cont %3d: ACPF failed.\n', ...
                    scen_name, h, cont_idx);

                % -------------------------
                % Store NaN train row
                % -------------------------
                row_train = nan(1, 2 + 4*nb);
                row_train(1) = sample_id;
                row_train(2) = cont_idx;
                train_rows{end+1,1} = row_train;

                % -------------------------
                % Store NaN dispatch row
                % -------------------------
                row_disp = nan(1, 2 + 2*ng);
                row_disp(1) = sample_id;
                row_disp(2) = cont_idx;
                dispatch_rows{end+1,1} = row_disp;

                % -------------------------
                % Store meta row
                % -------------------------
                meta_rows(end+1,:) = { ...
                    sample_id, scen_name, h, cont_idx, ...
                    alpha, Pavail_wind, Pavail_solar, ...
                    r_opf.success, r_pf.success};

                % -------------------------
                % Store violation row also for failed PF
                % -------------------------
                violation_rows(end+1,:) = { ...
                    sample_id, scen_name, h, cont_idx, ...
                    NaN, NaN, NaN, NaN, NaN, ...
                    NaN, NaN, NaN, NaN, NaN, NaN, ...
                    NaN, r_pf.success};

                continue;
            end

            %% ------------------------------------------------
            % Identify voltage and branch violations
            %% ------------------------------------------------
            vm   = r_pf.bus(:, VM);
            vmin = r_pf.bus(:, VMIN);
            vmax = r_pf.bus(:, VMAX);

            under = vm < (vmin - 1e-9);
            over  = vm > (vmax + 1e-9);
            viol_v = under | over;

            num_vunder = sum(under);
            num_vover  = sum(over);
            num_vviol  = sum(viol_v);

            if any(viol_v)
                vdev = max([vmin - vm, vm - vmax, zeros(nb,1)], [], 2);
                [worst_vdev, worst_v_idx] = max(vdev);
                worst_vm   = vm(worst_v_idx);
                worst_vbus = r_pf.bus(worst_v_idx, BUS_I);
            else
                [worst_vm, worst_v_idx] = min(vm);
                worst_vbus = r_pf.bus(worst_v_idx, BUS_I);
                worst_vdev = 0;
            end

            % -------------------------
            % Branch thermal violations
            % -------------------------
            Sf = hypot(r_pf.branch(:, PF), r_pf.branch(:, QF));
            St = hypot(r_pf.branch(:, PT), r_pf.branch(:, QT));
            Sbr = max(Sf, St);

            rateA = r_pf.branch(:, RATE_A);
            in_service = r_pf.branch(:, BR_STATUS) > 0;

            viol_br = in_service & (Sbr > rateA + 1e-6);
            num_brviol = sum(viol_br);

            overload_pct = zeros(nbr,1);
            valid = in_service & (rateA > 0);
            overload_pct(valid) = max(0, (Sbr(valid) - rateA(valid)) ./ rateA(valid) * 100);

            if any(viol_br)
                [worst_ov_pct, worst_br_idx] = max(overload_pct);
                worst_flow_mva  = Sbr(worst_br_idx);
                worst_limit_mva = rateA(worst_br_idx);
            else
                worst_ov_pct    = 0;
                worst_br_idx    = NaN;
                worst_flow_mva  = NaN;
                worst_limit_mva = NaN;
            end

            total_viol = num_vviol + num_brviol;

            %% ------------------------------------------------
            % Store PF solution row
            %% ------------------------------------------------
            row_train = zeros(1, 2 + 4*nb);
            row_train(1) = sample_id;
            row_train(2) = cont_idx;

            col = 3;
            for b = 1:nb
                row_train(col) = r_pf.bus(b, PD); col = col + 1;
                row_train(col) = r_pf.bus(b, QD); col = col + 1;
                row_train(col) = r_pf.bus(b, VM); col = col + 1;
                row_train(col) = r_pf.bus(b, VA); col = col + 1;
            end
            train_rows{end+1,1} = row_train;

            row_disp = zeros(1, 2 + 2*ng);
            row_disp(1) = sample_id;
            row_disp(2) = cont_idx;

            col = 3;
            for g = 1:ng
                row_disp(col) = r_pf.gen(g, PG); col = col + 1;
                row_disp(col) = r_pf.gen(g, QG); col = col + 1;
            end
            dispatch_rows{end+1,1} = row_disp;

            meta_rows(end+1,:) = { ...
                sample_id, scen_name, h, cont_idx, ...
                alpha, Pavail_wind, Pavail_solar, ...
                r_opf.success, r_pf.success};

            violation_rows(end+1,:) = { ...
                sample_id, scen_name, h, cont_idx, ...
                num_brviol, worst_ov_pct, worst_br_idx, worst_flow_mva, worst_limit_mva, ...
                num_vviol, num_vunder, num_vover, worst_vm, worst_vbus, worst_vdev, ...
                total_viol, r_pf.success};
        end

    end
end

%% ============================================================
% Convert to tables
%% ============================================================
train_varnames = cell(1, 2 + 4*nb);
train_varnames{1} = 'sample_id';
train_varnames{2} = 'contingency_index';

col = 3;
for b = 1:nb
    train_varnames{col} = sprintf('Pload_%d', b); col = col + 1;
    train_varnames{col} = sprintf('Qload_%d', b); col = col + 1;
    train_varnames{col} = sprintf('V_%d', b);     col = col + 1;
    train_varnames{col} = sprintf('delta_%d', b); col = col + 1;
end

train_mat = cell2mat(train_rows);
train = array2table(train_mat, 'VariableNames', train_varnames);

dispatch_varnames = cell(1, 2 + 2*ng);
dispatch_varnames{1} = 'sample_id';
dispatch_varnames{2} = 'contingency_index';

col = 3;
for g = 1:ng
    dispatch_varnames{col} = sprintf('Pgen_%d', g); col = col + 1;
    dispatch_varnames{col} = sprintf('Qgen_%d', g); col = col + 1;
end

dispatch_mat = cell2mat(dispatch_rows);
train_dispatch = array2table(dispatch_mat, 'VariableNames', dispatch_varnames);

sample_meta = cell2table(meta_rows, ...
    'VariableNames', { ...
    'sample_id','scenario_name','hour','contingency_index', ...
    'load_multiplier','wind_MW','solar_MW','opf_success','pf_success'});

fprintf('Number of violation rows stored = %d\n', size(violation_rows,1));

violation_summary = cell2table(violation_rows, ...
    'VariableNames', { ...
    'sample_id', 'scenario_name', 'hour', 'contingency_index', ...
    'NumBranchViolations', 'WorstOverload_pct', 'WorstOverloadBranch', ...
    'WorstFlow_MVA', 'Limit_MVA', ...
    'NumVoltageViolations', 'NumUndervoltage', 'NumOvervoltage', ...
    'WorstVM_pu', 'WorstVBus', 'WorstVDev_pu', ...
    'TotalViolations', 'pf_success'});

%% ============================================================
% Export to Excel
%% ============================================================
outfile = 'case118_contingency_pf_dataset_8scen_8cont.xlsx';

writetable(case118_branch_data, outfile, 'Sheet', 'case118_branch_data');
writetable(case118_bus_type,    outfile, 'Sheet', 'case118_bus_type');
writetable(train,               outfile, 'Sheet', 'train');
writetable(train_dispatch,      outfile, 'Sheet', 'train_dispatch');
writetable(sample_meta,         outfile, 'Sheet', 'sample_meta');
writetable(violation_summary, outfile, 'Sheet', 'violation_summary');

fprintf('\nDataset saved to %s\n', outfile);
fprintf('Total samples stored = %d\n', height(train));





%% ============================================================
% Plot seasonal curves and max/min curves
%% ============================================================

hours = (1:24)';

%% ------------------------------------------------------------
% 1) Seasonal load curves
%% ------------------------------------------------------------
figure;
hold on; grid on; box on;

plot(hours, load_winter, '-b', 'LineWidth', 2.5);
plot(hours, load_spring, '-g', 'LineWidth', 2.5);
plot(hours, load_summer, '-r', 'LineWidth', 2.5);
plot(hours, load_fall,   '-m', 'LineWidth', 2.5);

xlabel('Hour');
ylabel('Load scale factor');
title('Seasonal Load Curves');
legend('Winter', 'Spring', 'Summer', 'Fall', 'Location', 'best');
xlim([1 24]);
xticks(1:24);

%% ------------------------------------------------------------
% 2) Seasonal wind curves
%% ------------------------------------------------------------
figure;
hold on; grid on; box on;

plot(hours, wind_winter, '-b', 'LineWidth', 2.5);
plot(hours, wind_spring, '-g', 'LineWidth', 2.5);
plot(hours, wind_summer, '-r', 'LineWidth', 2.5);
plot(hours, wind_fall,   '-m', 'LineWidth', 2.5);

xlabel('Hour');
ylabel('Wind capacity factor');
title('Seasonal Wind Curves');
legend('Winter', 'Spring', 'Summer', 'Fall', 'Location', 'best');
xlim([1 24]);
xticks(1:24);

%% ------------------------------------------------------------
% 3) Seasonal solar curves
%% ------------------------------------------------------------
figure;
hold on; grid on; box on;

plot(hours, pv_winter, '-b', 'LineWidth', 2.5);
plot(hours, pv_spring, '-g', 'LineWidth', 2.5);
plot(hours, pv_summer, '-r', 'LineWidth', 2.5);
plot(hours, pv_fall,   '-m', 'LineWidth', 2.5);

xlabel('Hour');
ylabel('Solar capacity factor');
title('Seasonal Solar Curves');
legend('Winter', 'Spring', 'Summer', 'Fall', 'Location', 'best');
xlim([1 24]);
xticks(1:24);

%% ------------------------------------------------------------
% 4) Max and min curves for load, wind, solar
%% ------------------------------------------------------------
figure;
tiledlayout(3,1);

% -------------------------
% Load min/max
% -------------------------
nexttile;
hold on; grid on; box on;
plot(hours, Lmax, '-r', 'LineWidth', 2.5);
plot(hours, Lmin, '--b', 'LineWidth', 2.5);
xlabel('Hour');
ylabel('Load scale factor');
title('Load Max and Min Curves');
legend('L_{max}', 'L_{min}', 'Location', 'best');
xlim([1 24]);
xticks(1:24);

% -------------------------
% Wind min/max
% -------------------------
nexttile;
hold on; grid on; box on;
plot(hours, Wmax, '-r', 'LineWidth', 2.5);
plot(hours, Wmin, '--b', 'LineWidth', 2.5);
xlabel('Hour');
ylabel('Wind capacity factor');
title('Wind Max and Min Curves');
legend('W_{max}', 'W_{min}', 'Location', 'best');
xlim([1 24]);
xticks(1:24);

% -------------------------
% Solar min/max
% -------------------------
nexttile;
hold on; grid on; box on;
plot(hours, PVmax, '-r', 'LineWidth', 2.5);
plot(hours, PVmin, '--b', 'LineWidth', 2.5);
xlabel('Hour');
ylabel('Solar capacity factor');
title('Solar Max and Min Curves');
legend('PV_{max}', 'PV_{min}', 'Location', 'best');
xlim([1 24]);
xticks(1:24);

%% ------------------------------------------------------------
% Optional: save figures
%% ------------------------------------------------------------
saveas(figure(1), 'Seasonal_Load_Curves.png');
saveas(figure(2), 'Seasonal_Wind_Curves.png');
saveas(figure(3), 'Seasonal_Solar_Curves.png');
saveas(figure(4), 'MaxMin_Load_Wind_Solar_Curves.png');