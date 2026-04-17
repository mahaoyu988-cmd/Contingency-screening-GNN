define_constants;
mpopt_ac = mpoption('verbose', 0, 'out.all', 0);
mpopt_dc = mpoption('verbose', 0, 'out.all', 0);

%% ============================================================
% INPUTS REQUIRED BEFORE THIS SCRIPT
% ------------------------------------------------------------
% You should already have:
%   results24{h}   = hourly ACOPF result for hour h, h = 1..24
%   mpc0           = base case with RATE_A already assigned
%   load profiles and renewable replacement already embedded in hourly OPF
%
% This script performs, for each hour:
%   1) start from ACOPF result
%   2) keep only PG as fixed pre-contingency dispatch
%   3) use VM and VA as ACPF initial guess
%   4) do NOT carry OPF VG or QG into ACPF
%   5) run N-1 contingency screening on every meshed line using ACPF and DCPF
%   6) output ACPF and DCPF result tables for outages causing violations
%% ============================================================

tolS = 1e-6;
monitor_threshold = 0.50;   % keep branches with loading > 50%

nHours = 24;

% containers
MonitoredBranches = cell(nHours,1);
ResultsAC_hour = cell(nHours,1);
ResultsDC_hour = cell(nHours,1);

for h = 1:nHours
    fprintf('\n============================================================\n');
    fprintf('Hour %d contingency screening\n', h);
    fprintf('============================================================\n');

    if isempty(results24{h}) || ~results24{h}.success
        fprintf('Hour %d skipped: OPF did not converge.\n', h);
        ResultsAC_hour{h} = table();
        ResultsDC_hour{h} = table();
        MonitoredBranches{h} = table();
        continue;
    end

    %% --------------------------------------------------------
    % 1) Build ACPF and DCPF starting cases from ACOPF result
    %% --------------------------------------------------------
    ac_opf = results24{h};

    % -------------------------
    % ACPF starting case
    % -------------------------
    % Start from the original hourly solved case structure
    ac118_h = ac_opf;

    % Keep active-power dispatch from ACOPF
    ac118_h.gen(:, PG) = ac_opf.gen(:, PG);

    % Fix generator active power for screening
    % (strict no-redispatch formulation)
    ac118_h.gen(:, PMAX) = ac_opf.gen(:, PG);
    ac118_h.gen(:, PMIN) = ac_opf.gen(:, PG);

    % Do NOT carry OPF QG as a fixed value
    % Use it only as optional initial guess or reset to zero
    ac118_h.gen(:, QG) = 0;

    % Do NOT carry OPF VG
    % Use default generator voltage setpoint = 1.0 pu
    ac118_h.gen(:, VG) = 1.0;

    % Use OPF voltages/angles as initial guess for AC PF
    ac118_h.bus(:, VM) = ac_opf.bus(:, VM);
    ac118_h.bus(:, VA) = ac_opf.bus(:, VA);

    % -------------------------
    % DCPF starting case
    % -------------------------
    dc118_h = ac_opf;

    % Keep same active-power dispatch
    dc118_h.gen(:, PG) = ac_opf.gen(:, PG);

    % Fix generator active power
    dc118_h.gen(:, PMAX) = ac_opf.gen(:, PG);
    dc118_h.gen(:, PMIN) = ac_opf.gen(:, PG);

    % Q and voltage magnitude are not used in DC PF
    dc118_h.gen(:, QG) = 0;
    dc118_h.bus(:, VM) = 1.0;
    dc118_h.bus(:, VA) = 0.0;

    %% --------------------------------------------------------
    % 2) Pre-contingency monitored branches (>50% loaded)
    %% --------------------------------------------------------
    Sf0 = hypot(ac_opf.branch(:, PF), ac_opf.branch(:, QF));
    St0 = hypot(ac_opf.branch(:, PT), ac_opf.branch(:, QT));
    S0  = max(Sf0, St0);
    % S0 = calc_branch_flow_avg(ac_opf.branch);

    RateA0 = ac_opf.branch(:, RATE_A);
    RateA0(RateA0 <= 0) = Inf;

    loading0 = S0 ./ RateA0;
    in0 = ac_opf.branch(:, BR_STATUS) > 0;

    monitored_mask = in0 & (loading0 >= monitor_threshold);
    monitored_idx  = find(monitored_mask);

    MonitoredBranches{h} = table( ...
        monitored_idx, ...
        ac_opf.branch(monitored_idx, F_BUS), ...
        ac_opf.branch(monitored_idx, T_BUS), ...
        S0(monitored_idx), ...
        RateA0(monitored_idx), ...
        100*loading0(monitored_idx), ...
        'VariableNames', {'BranchIdx','FromBus','ToBus','Flow_MVA','Limit_MVA','Loading_pct'});

    fprintf('Hour %d: %d monitored branches loaded above 50%%.\n', ...
        h, numel(monitored_idx));

    %% --------------------------------------------------------
    % 3) Identify meshed outage candidates
    %% --------------------------------------------------------
    meshed_branch_idx = get_meshed_branch_idx(ac118_h);

    fprintf('Hour %d: %d meshed outage candidates.\n', h, numel(meshed_branch_idx));

    %% --------------------------------------------------------
    % 4) Run ACPF contingency screening
    %% --------------------------------------------------------
    contAC = init_cont_struct(numel(meshed_branch_idx));

    for i = 1:numel(meshed_branch_idx)
        br_out = meshed_branch_idx(i);

        mpc = ac118_h;
        mpc.branch(br_out, BR_STATUS) = 0;

        r = runpf(mpc, mpopt_ac);

        contAC.branch_idx(i) = br_out;
        contAC.fbus(i) = mpc.branch(br_out, F_BUS);
        contAC.tbus(i) = mpc.branch(br_out, T_BUS);

        if ~r.success
            contAC.pf_converged(i)    = false;
            contAC.num_branch_viol(i) = Inf;
            contAC.max_over_pct(i)    = Inf;
            contAC.max_over_line(i)   = NaN;
            contAC.max_flow_mva(i)    = NaN;
            contAC.limit_mva(i)       = NaN;

            contAC.num_vunder(i)      = NaN;
            contAC.num_vover(i)       = NaN;
            contAC.num_vviol(i)       = Inf;
            contAC.worst_v_bus(i)     = NaN;
            contAC.worst_vm(i)        = NaN;
            contAC.worst_vmin(i)      = NaN;
            contAC.worst_vmax(i)      = NaN;
            contAC.worst_vdev_pu(i)   = Inf;
            continue;
        end

        contAC.pf_converged(i) = true;

        br = r.branch;
        in = br(:, BR_STATUS) > 0;

        Sf = hypot(br(:, PF), br(:, QF));
        St = hypot(br(:, PT), br(:, QT));
        S  = max(Sf, St);
        % S = calc_branch_flow_avg(br);

        rateA = br(:, RATE_A);
        rateA(rateA <= 0) = Inf;

        % thermal violations only on monitored branches (>50% in base case)
        viol_mask = false(size(S));
        viol_mask(monitored_idx) = in(monitored_idx) & (S(monitored_idx) > rateA(monitored_idx) + tolS);

        contAC.num_branch_viol(i) = sum(viol_mask);

        over_ratio = zeros(size(S));
        over_ratio(monitored_idx) = S(monitored_idx) ./ rateA(monitored_idx);

        [worst_ratio, worst_line] = max(over_ratio);
        worst_pct = max(0, (worst_ratio - 1) * 100);

        contAC.max_over_pct(i)  = worst_pct;
        contAC.max_over_line(i) = worst_line;
        contAC.max_flow_mva(i)  = S(worst_line);
        contAC.limit_mva(i)     = rateA(worst_line);

        % voltage violations on all buses
        vm   = r.bus(:, VM);
        vmin = r.bus(:, VMIN);
        vmax = r.bus(:, VMAX);

        under = vm < (vmin - 1e-9);
        over  = vm > (vmax + 1e-9);
        viol  = under | over;

        contAC.num_vunder(i) = sum(under);
        contAC.num_vover(i)  = sum(over);
        contAC.num_vviol(i)  = sum(viol);

        if any(viol)
            dev = max(max(vmin - vm, 0), max(vm - vmax, 0));
            [dmax, bidx] = max(dev);

            contAC.worst_vdev_pu(i) = dmax;
            contAC.worst_v_bus(i)   = r.bus(bidx, BUS_I);
            contAC.worst_vm(i)      = vm(bidx);
            contAC.worst_vmin(i)    = vmin(bidx);
            contAC.worst_vmax(i)    = vmax(bidx);
        else
            contAC.worst_vdev_pu(i) = 0;
            contAC.worst_v_bus(i)   = NaN;
            contAC.worst_vm(i)      = NaN;
            contAC.worst_vmin(i)    = NaN;
            contAC.worst_vmax(i)    = NaN;
        end
    end

    resultsAC = rank_results_combined(contAC);

    keepAC = (~resultsAC.PF_Converged) | ...
             (resultsAC.NumBranchViolations > 0) | ...
             (resultsAC.NumVoltageViolations > 0);

    resultsAC = resultsAC(keepAC, :);
    ResultsAC_hour{h} = resultsAC;

    %% --------------------------------------------------------
    % 5) Run DCPF contingency screening
    %% --------------------------------------------------------
    contDC = init_cont_struct(numel(meshed_branch_idx));

    for i = 1:numel(meshed_branch_idx)
        br_out = meshed_branch_idx(i);

        mpc = dc118_h;
        mpc.branch(br_out, BR_STATUS) = 0;

        r = rundcpf(mpc, mpopt_dc);

        contDC.branch_idx(i) = br_out;
        contDC.fbus(i) = mpc.branch(br_out, F_BUS);
        contDC.tbus(i) = mpc.branch(br_out, T_BUS);

        if ~r.success
            contDC.pf_converged(i)    = false;
            contDC.num_branch_viol(i) = Inf;
            contDC.max_over_pct(i)    = Inf;
            contDC.max_over_line(i)   = NaN;
            contDC.max_flow_mva(i)    = NaN;
            contDC.limit_mva(i)       = NaN;

            contDC.num_vunder(i)      = NaN;
            contDC.num_vover(i)       = NaN;
            contDC.num_vviol(i)       = NaN;
            contDC.worst_v_bus(i)     = NaN;
            contDC.worst_vm(i)        = NaN;
            contDC.worst_vmin(i)      = NaN;
            contDC.worst_vmax(i)      = NaN;
            contDC.worst_vdev_pu(i)   = NaN;
            continue;
        end

        contDC.pf_converged(i) = true;

        br = r.branch;
        in = br(:, BR_STATUS) > 0;

        Sf = hypot(br(:, PF), br(:, QF));
        St = hypot(br(:, PT), br(:, QT));
        S  = max(Sf, St);
        % S = calc_branch_flow_avg(br);

        rateA = br(:, RATE_A);
        rateA(rateA <= 0) = Inf;

        % thermal violations only on monitored branches
        viol_mask = false(size(S));
        viol_mask(monitored_idx) = in(monitored_idx) & (S(monitored_idx) > rateA(monitored_idx) + tolS);

        contDC.num_branch_viol(i) = sum(viol_mask);

        over_ratio = zeros(size(S));
        over_ratio(monitored_idx) = S(monitored_idx) ./ rateA(monitored_idx);

        [worst_ratio, worst_line] = max(over_ratio);
        worst_pct = max(0, (worst_ratio - 1) * 100);

        contDC.max_over_pct(i)  = worst_pct;
        contDC.max_over_line(i) = worst_line;
        contDC.max_flow_mva(i)  = S(worst_line);
        contDC.limit_mva(i)     = rateA(worst_line);

        % DCPF has no meaningful voltage-magnitude screening
        contDC.num_vunder(i)    = 0;
        contDC.num_vover(i)     = 0;
        contDC.num_vviol(i)     = 0;
        contDC.worst_v_bus(i)   = NaN;
        contDC.worst_vm(i)      = NaN;
        contDC.worst_vmin(i)    = NaN;
        contDC.worst_vmax(i)    = NaN;
        contDC.worst_vdev_pu(i) = 0;
    end

    resultsDC = rank_results_combined(contDC);

    keepDC = (~resultsDC.PF_Converged) | ...
             (resultsDC.NumBranchViolations > 0) | ...
             (resultsDC.NumVoltageViolations > 0);

    resultsDC = resultsDC(keepDC, :);
    ResultsDC_hour{h} = resultsDC;

    %% --------------------------------------------------------
    % 6) Display summary
    %% --------------------------------------------------------
    fprintf('\nHour %d ACPF: outages causing violations = %d\n', h, height(resultsAC));
    disp(resultsAC);

    fprintf('\nHour %d DCPF: outages causing violations = %d\n', h, height(resultsDC));
    disp(resultsDC);
end

%% ============================================================
% 7) Export to Excel
%% ============================================================
outfile = 'Hourly_Contingency_Screening_24h.xlsx';

for h = 1:nHours
    writetable(MonitoredBranches{h}, outfile, 'Sheet', sprintf('Hour%02d_Monitored', h));
    writetable(ResultsAC_hour{h},  outfile, 'Sheet', sprintf('Hour%02d_ACPF', h));
    writetable(ResultsDC_hour{h},  outfile, 'Sheet', sprintf('Hour%02d_DCPF', h));
end

fprintf('\nSaved hourly screening results to %s\n', outfile);

%% ============================================================
% Helper functions
%% ============================================================
% function Sbr = calc_branch_flow_avg(branch)
%     define_constants;
%     Pbr = (branch(:, PF) - branch(:, PT)) / 2;
%     Qbr = (branch(:, QF) - branch(:, QT)) / 2;
%     Sbr = hypot(Pbr, Qbr);
% end



function meshed_branch_idx = get_meshed_branch_idx(mpc)
    define_constants;
    fbus = mpc.branch(:, F_BUS);
    tbus = mpc.branch(:, T_BUS);
    status = mpc.branch(:, BR_STATUS) > 0;

    edge_idx_all = find(status);
    G = graph(fbus(status), tbus(status));

    isBridge = false(numel(edge_idx_all), 1);
    baseComp = conncomp(G);
    nComp0 = max(baseComp);

    for k = 1:numel(edge_idx_all)
        keep = true(numel(edge_idx_all), 1);
        keep(k) = false;

        Gk = graph(fbus(edge_idx_all(keep)), tbus(edge_idx_all(keep)));
        ck = conncomp(Gk);
        nCompK = max(ck);

        if nCompK > nComp0
            isBridge(k) = true;
        end
    end

    meshed_edge_local = find(~isBridge);
    meshed_branch_idx = edge_idx_all(meshed_edge_local);
end

function cont = init_cont_struct(nC)
    cont.branch_idx        = zeros(nC, 1);
    cont.fbus              = zeros(nC, 1);
    cont.tbus              = zeros(nC, 1);
    cont.pf_converged      = false(nC, 1);

    cont.num_branch_viol   = zeros(nC, 1);
    cont.max_over_pct      = zeros(nC, 1);
    cont.max_over_line     = zeros(nC, 1);
    cont.max_flow_mva      = zeros(nC, 1);
    cont.limit_mva         = zeros(nC, 1);

    cont.num_vunder        = zeros(nC, 1);
    cont.num_vover         = zeros(nC, 1);
    cont.num_vviol         = zeros(nC, 1);
    cont.worst_v_bus       = zeros(nC, 1);
    cont.worst_vm          = zeros(nC, 1);
    cont.worst_vmin        = zeros(nC, 1);
    cont.worst_vmax        = zeros(nC, 1);
    cont.worst_vdev_pu     = zeros(nC, 1);
end

function results = rank_results_combined(cont)
    sortMat = [ ...
        double(~cont.pf_converged), ...
        -cont.num_branch_viol, ...
        -cont.num_vviol, ...
        -cont.max_over_pct, ...
        cont.worst_vm ...
    ];

    [~, order] = sortrows(sortMat, [1 2 3 4 5]);

    results = table( ...
        cont.branch_idx(order), ...
        cont.fbus(order), ...
        cont.tbus(order), ...
        cont.pf_converged(order), ...
        cont.num_branch_viol(order), ...
        cont.max_over_pct(order), ...
        cont.max_over_line(order), ...
        cont.max_flow_mva(order), ...
        cont.limit_mva(order), ...
        cont.num_vunder(order), ...
        cont.num_vover(order), ...
        cont.num_vviol(order), ...
        cont.worst_vm(order), ...
        cont.worst_vdev_pu(order), ...
        'VariableNames', { ...
            'OutagedBranch','FromBus','ToBus','PF_Converged', ...
            'NumBranchViolations','WorstOverload_pct', ...
            'WorstOverloadLine','WorstFlow_MVA','Limit_MVA', ...
            'NumUndervoltageViolations','NumOvervoltageViolations', ...
            'NumVoltageViolations','WorstVM_pu','WorstVDev_pu'} );
end

%% ============================================================
% Count number of outages causing each type of violation
%% ============================================================

%% ============================================================
% Count number of outages causing each violation type (AC vs DC)
%% ============================================================

nHours = 24;

num_outage_branchviol_AC = zeros(nHours,1);
num_outage_branchviol_DC = zeros(nHours,1);
num_outage_undervolt_AC  = zeros(nHours,1);
num_outage_undervolt_DC  = zeros(nHours,1);   % DCPF cannot capture voltage magnitude violations

for h = 1:nHours
    % ----- AC results -----
    if ~isempty(ResultsAC_hour{h}) && height(ResultsAC_hour{h}) > 0
        TAC = ResultsAC_hour{h};

        % number of outages causing at least one branch thermal violation
        num_outage_branchviol_AC(h) = sum(TAC.NumBranchViolations > 0);

        % number of outages causing at least one undervoltage violation
        % This requires NumUndervoltageViolations to be included in the AC table
        if ismember('NumUndervoltageViolations', TAC.Properties.VariableNames)
            num_outage_undervolt_AC(h) = sum(TAC.NumUndervoltageViolations > 0);
        else
            warning('NumUndervoltageViolations not found in ResultsAC_hour{%d}.', h);
        end
    end

    % ----- DC results -----
    if ~isempty(ResultsDC_hour{h}) && height(ResultsDC_hour{h}) > 0
        TDC = ResultsDC_hour{h};

        % number of outages causing at least one branch thermal violation
        num_outage_branchviol_DC(h) = sum(TDC.NumBranchViolations > 0);

        % DCPF has no meaningful undervoltage results
        num_outage_undervolt_DC(h) = 0;
    end
end

%% ============================================================
% Grouped bar plot: AC vs DC comparison
%% ============================================================

figure;
bar_data = [ ...
    num_outage_branchviol_AC, ...
    num_outage_branchviol_DC, ...
    num_outage_undervolt_AC, ...
    num_outage_undervolt_DC];

bar(1:nHours, bar_data, 'grouped');

grid on;
xlabel('Hour');
ylabel('Number of outages');
title('ACPF vs DCPF: outages causing branch thermal and undervoltage violations');
legend('AC Branch Thermal', 'DC Branch Thermal', ...
       'AC Undervoltage', 'DC Undervoltage', ...
       'Location', 'best');