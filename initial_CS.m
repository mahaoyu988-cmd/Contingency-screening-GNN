define_constants;
verbose = 0;
mpopt = mpoption('verbose', 0, 'out.all', 0);
mpc = loadcase('case118.m'); % load the power system data

%give load scale factor to make violations more visable
alpha = 1.50;              % 20% load increase (use <1 for decrease)
mpc.bus(:, PD) = alpha * mpc.bus(:, PD);
mpc.bus(:, QD) = alpha * mpc.bus(:, QD);

%% Define branch thermal limits
base_kv = mpc.bus(:, BASE_KV);
nbr = size(mpc.branch, 1);
Smax = zeros(nbr, 1);

for l = 1:nbr
    fb = mpc.branch(l, F_BUS);       % FROM bus
    kv = base_kv(fb);                % Use FROM-bus voltage level

    if abs(kv - 345) < 1e-3
        Smax(l) = 700;   % 345 kV
    elseif abs(kv - 161) < 1e-3
        Smax(l) = 220;   % 161 kV
    else
        Smax(l) = 180;   % assume 138 kV
    end
end

mpc.branch(:, RATE_A) = Smax;

ac_opf = runopf(mpc);
dc_opf = rundcopf(mpc);

tolS = 1e-6;     % tolerance for thermal violation
tolV = 1e-6;     % tolerance for voltage violation

% === AC branch apparent power (per end), then max end ===
Pac_from = ac_opf.branch(:, PF);
Qac_from = ac_opf.branch(:, QF);
Pac_to   = ac_opf.branch(:, PT);
Qac_to   = ac_opf.branch(:, QT);

Sac_fr = hypot(Pac_from, Qac_from);      % sqrt(P^2 + Q^2)
Sac_to = hypot(Pac_to,   Qac_to);
Sbrac  = max(Sac_fr, Sac_to);

RateA_ac = ac_opf.branch(:, RATE_A);     % should match mpc.branch(:,RATE_A)

overload_ac = Sbrac - RateA_ac;
acvio_idx = find(overload_ac > tolS);

% === DC branch "S" (Q is (usually) ~0), then max end ===
Pdc_from = dc_opf.branch(:, PF);
Qdc_from = dc_opf.branch(:, QF);         % typically 0 in DCOPF results
Pdc_to   = dc_opf.branch(:, PT);
Qdc_to   = dc_opf.branch(:, QT);

Sdc_fr = hypot(Pdc_from, Qdc_from);
Sdc_to = hypot(Pdc_to,   Qdc_to);
Sbrdc  = max(Sdc_fr, Sdc_to);

RateA_dc = dc_opf.branch(:, RATE_A);

overload_dc = Sbrdc - RateA_dc;
dcvio_idx = find(overload_dc > tolS);

%% -----------------------------
%  Print branch thermal violations (AC & DC)
% -----------------------------
fprintf('\n================ Branch Thermal Violations (ACOPF) ================\n');
if isempty(acvio_idx)
    fprintf('No AC thermal violations.\n');
else
    fprintf('Idx   From  To     SbrAC   RateA   Overload\n');
    for k = 1:length(acvio_idx)
        i = acvio_idx(k);
        fb = ac_opf.branch(i, F_BUS);
        tb = ac_opf.branch(i, T_BUS);
        fprintf('%3d   %4d  %4d   %8.2f %8.2f %9.2f\n', ...
            i, fb, tb, Sbrac(i), RateA_ac(i), overload_ac(i));
    end
end

fprintf('\n================ Branch Thermal Violations (DCOPF) ================\n');
if isempty(dcvio_idx)
    fprintf('No DC thermal violations.\n');
else
    fprintf('Idx   From  To     SbrDC   RateA   Overload\n');
    for k = 1:length(dcvio_idx)
        i = dcvio_idx(k);
        fb = dc_opf.branch(i, F_BUS);
        tb = dc_opf.branch(i, T_BUS);
        fprintf('%3d   %4d  %4d   %8.2f %8.2f %9.2f\n', ...
            i, fb, tb, Sbrdc(i), RateA_dc(i), overload_dc(i));
    end
end

%% -----------------------------
%  Bus voltage violations (ACOPF; optional DCOPF too)
% -----------------------------
Vm_ac   = ac_opf.bus(:, VM);
Vmax_ac = ac_opf.bus(:, VMAX);
Vmin_ac = ac_opf.bus(:, VMIN);

vlow_ac  = find(Vm_ac < Vmin_ac - tolV);
vhigh_ac = find(Vm_ac > Vmax_ac + tolV);
vviol_ac = unique([vlow_ac; vhigh_ac]);

fprintf('\n================ Bus Voltage Violations (ACOPF) ===================\n');
if isempty(vviol_ac)
    fprintf('No AC voltage violations.\n');
else
    fprintf('Bus   Vm(pu)   Vmin   Vmax   Violation\n');
    for k = 1:length(vviol_ac)
        b = vviol_ac(k);
        viol_str = "";
        if Vm_ac(b) < Vmin_ac(b) - tolV, viol_str = viol_str + "LOW "; end
        if Vm_ac(b) > Vmax_ac(b) + tolV, viol_str = viol_str + "HIGH"; end
        fprintf('%3d   %6.3f  %6.3f %6.3f   %s\n', ...
            b, Vm_ac(b), Vmin_ac(b), Vmax_ac(b), viol_str);
    end
end

% %DC bus voltage check (often Vm ~ 1.0 everywhere in DCOPF)
% Vm_dc   = dc_opf.bus(:, VM);
% Vmax_dc = dc_opf.bus(:, VMAX);
% Vmin_dc = dc_opf.bus(:, VMIN);
% vviol_dc = find((Vm_dc < Vmin_dc - tolV) | (Vm_dc > Vmax_dc + tolV));
% 
% fprintf('\n================ Bus Voltage Violations (DCOPF) ===================\n');
% if isempty(vviol_dc)
%     fprintf('No DC voltage violations (expected in most DCOPF cases).\n');
% else
%     fprintf('Bus   Vm(pu)   Vmin   Vmax\n');
%     for k = 1:length(vviol_dc)
%         b = vviol_dc(k);
%         fprintf('%3d   %6.3f  %6.3f %6.3f\n', b, Vm_dc(b), Vmin_dc(b), Vmax_dc(b));
%     end
% end


%% -----------------------------
%  FIGURE 1: Branch flows vs limits (grouped columns)
% -----------------------------
nbr = size(ac_opf.branch, 1);
br_idx = (1:nbr)';

% --- After you compute Sbrac, Sbrdc, and have ac_opf, dc_opf ---
RateA = ac_opf.branch(:, RATE_A);
br = (1:length(RateA)).';

figure; hold on; box on;

% Bar plot for RateA (thermal limits)
hb = bar(br, RateA, 1.0);  % 1.0 = full bar width
hb.FaceAlpha = 0.25;       % make it transparent
hb.EdgeColor = 'none';

% Line plots for branch apparent power
h1 = plot(br, Sbrac, 'LineWidth', 1.4);
h2 = plot(br, Sbrdc, 'LineWidth', 1.4);

grid on;
xlabel('Branch index');
ylabel('Apparent power |S| (MVA)');
title('Branch Apparent Power vs Thermal Limit');

% Optional: highlight only violations
tol = 1e-6;  % numerical tolerance (see note below)
ac_vio = find(Sbrac - RateA > tol);
dc_vio = find(Sbrdc - RateA > tol);

h3 = plot(ac_vio, Sbrac(ac_vio), 'o', 'MarkerSize', 6, 'LineWidth', 1.2);
h4 = plot(dc_vio, Sbrdc(dc_vio), 's', 'MarkerSize', 6, 'LineWidth', 1.2);

legend([h1 h2 hb h3 h4], ...
       {'Sbr (ACOPF)', 'Sbr (DCOPF)', 'RATE\_A', 'AC viol.', 'DC viol.'}, ...
       'Location', 'best');

%% -----------------------------
%  FIGURE 2: Bus voltages with Vmin/Vmax
% -----------------------------
nb = size(ac_opf.bus, 1);
bus_idx = (1:nb)';

figure('Name','Bus Voltage Profile with Limits');
plot(bus_idx, Vm_ac, 'LineWidth', 1.5); hold on;
plot(bus_idx, Vmin_ac, '--', 'LineWidth', 1.2);
plot(bus_idx, Vmax_ac, '--', 'LineWidth', 1.2);
grid on;
xlabel('Bus Number');
ylabel('Voltage Magnitude (pu)');
legend({'Vm (ACOPF)','Vmin','Vmax'}, 'Location','best');
title('ACOPF Bus Voltage Profile and Limits');

% Highlight violating buses (if any)
if ~isempty(vviol_ac)
    scatter(vviol_ac, Vm_ac(vviol_ac), 36, 'filled');
    legend({'Vm (ACOPF)','Vmin','Vmax','Violations'}, 'Location','best');
end

%%Initial dispatch results

gen_bus = mpc.gen(:, GEN_BUS);
Pg_ac   = ac_opf.gen(:, PG);   % MW
Pg_dc   = dc_opf.gen(:, PG);   % MW
Qg_ac   = ac_opf.gen(:, QG);   % MVAr (AC only meaningful)
Qg_dc   = dc_opf.gen(:, QG);   % often 0 / not meaningful in DC

dPg = Pg_ac - Pg_dc;
abs_dPg = abs(dPg);

% Build table
gen_dispatch_tbl = table( ...
    (1:length(gen_bus))', gen_bus, Pg_ac, Pg_dc, dPg, abs_dPg, Qg_ac, Qg_dc, ...
    'VariableNames', {'GenIdx','Bus','Pg_AC_MW','Pg_DC_MW','DeltaPg_ACminusDC_MW','AbsDeltaPg_MW','Qg_AC_MVAr','Qg_DC_MVAr'} );

% % Sort by largest absolute dispatch difference
% gen_dispatch_tbl_sorted = sortrows(gen_dispatch_tbl, 'AbsDeltaPg_MW', 'descend');

fprintf('\n================ Generator Dispatch (ACOPF vs DCOPF) ================\n');
disp(gen_dispatch_tbl);

% fprintf('\n===== Top 10 generators by |DeltaPg| =====\n');
% disp(gen_dispatch_tbl_sorted(1:min(10,height(gen_dispatch_tbl_sorted)), :));

%% -----------------------------
%  Total generation comparison
% -----------------------------
sumPg_ac = sum(Pg_ac);
sumPg_dc = sum(Pg_dc);
sumQg_ac = sum(Qg_ac);   % AC only meaningful

delta_total_MW = sumPg_ac - sumPg_dc;
delta_total_pct_vs_dc = 100 * delta_total_MW / max(sumPg_dc, 1e-9);

fprintf('\n================ Total Generation Comparison ================\n');
fprintf('Total PG (ACOPF): %.4f MW\n', sumPg_ac);
fprintf('Total PG (DCOPF): %.4f MW\n', sumPg_dc);
fprintf('Delta Total PG (AC-DC): %.4f MW (%.4f%% of DC)\n', delta_total_MW, delta_total_pct_vs_dc);
fprintf('Total QG (ACOPF): %.4f MVAr\n', sumQg_ac);

%% -----------------------------
%  Bar figure: generator dispatch comparison
% -----------------------------
ng = length(gen_bus);
x = 1:ng;

figure('Name','Generator Dispatch AC vs DC','Color','w');
bar(x, [Pg_ac Pg_dc], 'grouped'); grid on; box on;
xlabel('Generator Index');
ylabel('Dispatch PG (MW)');
title('Generator Dispatch Comparison: ACOPF vs DCOPF');
legend({'ACOPF PG','DCOPF PG'}, 'Location','best');

%%Perform Contingency Screening

% Build TWO separate cases with SAME limits
% -----------------------------
ac118 = mpc;    % start from same base with limits
dc118 = mpc;

% AC-initialized case
% -----------------------------
ac118.gen(:, PG) = ac_opf.gen(:, PG);
ac118.gen(:, QG) = ac_opf.gen(:, QG);

% Also initialize PV/slack voltage setpoints from OPF result
% (in MATPOWER, gen(:,VG) holds voltage setpoint for PV/slack)
ac118.gen(:, VG) = ac_opf.gen(:, VG);

% Optional but often helpful: also initialize bus Vm/Va
ac118.bus(:, VM) = ac_opf.bus(:, VM);
ac118.bus(:, VA) = ac_opf.bus(:, VA);

% DC-initialized case
% -----------------------------
dc118.gen(:, PG) = dc_opf.gen(:, PG);

% DCOPF doesn't solve QG; choose a consistent initialization:
% Option 1 (recommended): keep base-case QG and VG
% dc118.gen(:, QG) stays as in mpc.gen(:,QG)
% dc118.gen(:, VG) stays as in mpc.gen(:,VG)

% Option 2: use ACOPF QG & VG to help ACPF convergence from DC Pg
dc118.gen(:, QG) = dc_opf.gen(:, QG);
dc118.gen(:, VG) = dc_opf.gen(:, VG);

% Optional: bus voltage initialization (flat start or from ACOPF)
dc118.bus(:, VM) = dc_opf.bus(:, VM);
dc118.bus(:, VA) = dc_opf.bus(:, VA);

%% --- Identify meshed branches (non-bridges) ---
% Build an undirected graph of in-service branches using bus numbers.
fbus = mpc.branch(:, F_BUS);
tbus = mpc.branch(:, T_BUS);
status = mpc.branch(:, BR_STATUS) > 0;

% Candidate set is based on topology using current in-service lines
edge_idx_all = find(status);
G = graph(fbus(status), tbus(status));

% A branch is "meshed" if it is NOT a bridge (removing it does not disconnect the graph).
% We compute bridges by testing connectivity after removing each edge (OK for IEEE 118 size).
isBridge = false(numel(edge_idx_all), 1);

% Precompute baseline #connected components
baseComp = conncomp(G);
nComp0 = max(baseComp);

for k = 1:numel(edge_idx_all)
    e = edge_idx_all(k);

    % Remove this edge from the graph
    keep = true(numel(edge_idx_all), 1);
    keep(k) = false;

    Gk = graph(fbus(edge_idx_all(keep)), tbus(edge_idx_all(keep)));
    ck = conncomp(Gk);
    nCompK = max(ck);

    if nCompK > nComp0
        isBridge(k) = true;
    end
end

meshed_edge_local = find(~isBridge);          % indices within edge_idx_all
meshed_branch_idx = edge_idx_all(meshed_edge_local);  % indices in mpc.branch

fprintf('Total branches: %d\n', nbr);
fprintf('In-service branches: %d\n', numel(edge_idx_all));
fprintf('Meshed (non-bridge) candidate outages: %d\n', numel(meshed_branch_idx));

%% --- N-1 screening loop ---
nC = numel(meshed_branch_idx);

%% ============================================================
%  AC N-1 Screening (ACPF)
%% ============================================================
contAC = init_cont_struct(nC);

for i = 1:nC
    br_out = meshed_branch_idx(i);

    mpc = ac118;                     % start from AC-initialized case
    mpc.branch(br_out, BR_STATUS) = 0;

    r = runpf(mpc, mpopt);

    contAC.branch_idx(i) = br_out;
    contAC.fbus(i) = mpc.branch(br_out, F_BUS);
    contAC.tbus(i) = mpc.branch(br_out, T_BUS);

    if ~r.success
        contAC.pf_converged(i)  = false;
        contAC.max_over_pct(i)  = Inf;
        contAC.max_over_line(i) = NaN;
        contAC.max_flow_mva(i)  = NaN;
        contAC.limit_mva(i)     = NaN;

        % voltage fields (NEW)
        contAC.num_vunder(i)    = NaN;
        contAC.num_vover(i)     = NaN;
        contAC.num_vviol(i)     = Inf;   % treat as severe
        contAC.worst_v_bus(i)   = NaN;
        contAC.worst_vm(i)      = NaN;
        contAC.worst_vmin(i)    = NaN;
        contAC.worst_vmax(i)    = NaN;
        contAC.worst_vdev_pu(i) = Inf;
        continue;
    end

    contAC.pf_converged(i) = true;

    br = r.branch;
    in = br(:, BR_STATUS) > 0;

    Sf = hypot(br(:, PF), br(:, QF));
    St = hypot(br(:, PT), br(:, QT));
    S  = max(Sf, St);

    rateA = br(:, RATE_A);
    rateA(rateA <= 0) = Inf;

    over_ratio = zeros(size(S));
    over_ratio(in) = S(in) ./ rateA(in);

    [worst_ratio, worst_line] = max(over_ratio);
    worst_pct = max(0, (worst_ratio - 1) * 100);

    contAC.max_over_pct(i)  = worst_pct;
    contAC.max_over_line(i) = worst_line;
    contAC.max_flow_mva(i)  = S(worst_line);
    contAC.limit_mva(i)     = rateA(worst_line);


    % -----------------------------
    % Voltage violations (NEW)
    % -----------------------------
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
        dev = max(max(vmin - vm, 0), max(vm - vmax, 0));  % outside-band deviation
        [dmax, bidx] = max(dev);

        contAC.worst_vdev_pu(i) = dmax;
        contAC.worst_v_bus(i)   = r.bus(bidx, BUS_I);     % bus number
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

resultsAC = rank_results(contAC);
fprintf('\n================ TOP 10 CONTINGENCIES (THERMAL, ACPF on AC118) ================\n');
disp(resultsAC(1:min(10,height(resultsAC)), :));

resultsV_AC = rank_results_voltage(contAC);
fprintf('\n================ TOP 10 CONTINGENCIES (VOLTAGE, ACPF on AC118) ================\n');
disp(resultsV_AC(1:min(10,height(resultsV_AC)), :));

%% ============================================================
%  DC N-1 Screening (DCPF)
%% ============================================================
contDC = init_cont_struct(nC);

for i = 1:nC
    br_out = meshed_branch_idx(i);

    mpc = ac118;                     % start from DC-initialized case
    mpc.branch(br_out, BR_STATUS) = 0;

    r = rundcpf(mpc, mpopt);

    contDC.branch_idx(i) = br_out;
    contDC.fbus(i) = mpc.branch(br_out, F_BUS);
    contDC.tbus(i) = mpc.branch(br_out, T_BUS);

    if ~r.success
        contDC.pf_converged(i)  = false;
        contDC.max_over_pct(i)  = Inf;
        contDC.max_over_line(i) = NaN;
        contDC.max_flow_mva(i)  = NaN;
        contDC.limit_mva(i)     = NaN;
        continue;
    end

    contDC.pf_converged(i) = true;

    br = r.branch;
    in = br(:, BR_STATUS) > 0;

    % In DC, Q is ~0; but we keep hypot() for consistent code
    Sf = hypot(br(:, PF), br(:, QF));
    St = hypot(br(:, PT), br(:, QT));
    S  = max(Sf, St);

    rateA = br(:, RATE_A);
    rateA(rateA <= 0) = Inf;

    over_ratio = zeros(size(S));
    over_ratio(in) = S(in) ./ rateA(in);

    [worst_ratio, worst_line] = max(over_ratio);
    worst_pct = max(0, (worst_ratio - 1) * 100);

    contDC.max_over_pct(i)  = worst_pct;
    contDC.max_over_line(i) = worst_line;
    contDC.max_flow_mva(i)  = S(worst_line);
    contDC.limit_mva(i)     = rateA(worst_line);
end

resultsDC = rank_results(contDC);
fprintf('\n================ TOP 10 CONTINGENCIES (DCPF on DC118) ================\n');
disp(resultsDC(1:min(10,height(resultsDC)), :));

%% ===================== Helper functions ======================
function cont = init_cont_struct(nC)
    cont.branch_idx      = zeros(nC, 1);
    cont.fbus            = zeros(nC, 1);
    cont.tbus            = zeros(nC, 1);
    cont.pf_converged    = false(nC, 1);
    cont.max_over_pct    = zeros(nC, 1);
    cont.max_over_line   = zeros(nC, 1);
    cont.max_flow_mva    = zeros(nC, 1);
    cont.limit_mva       = zeros(nC, 1);
end

function results = rank_results(cont)
    % Sort descending by severity (Inf first)
    [~, order] = sort(cont.max_over_pct, 'descend');

    results = table( ...
        cont.branch_idx(order), cont.fbus(order), cont.tbus(order), ...
        cont.pf_converged(order), cont.max_over_pct(order), ...
        cont.max_over_line(order), cont.max_flow_mva(order), cont.limit_mva(order), ...
        'VariableNames', {'OutagedBranch','FromBus','ToBus','PF_Converged', ...
                          'WorstOverload_pct','WorstOverloadLine','WorstFlow_MVA','Limit_MVA'} );
end


function resultsV = rank_results_voltage(cont)
    % Rank by: (1) non-converged first, (2) num_vviol desc, (3) worst_vdev_pu desc
    % Robust against row/column mismatch or partially-initialized fields.

    % reference length (use branch_idx if available, otherwise fbus)
    if isfield(cont, 'branch_idx')
        n = numel(cont.branch_idx);
    else
        n = numel(cont.fbus);
    end

    % helper: make a field into n×1 column, padding/truncating as needed
    function v = col(field, fillval)
        if isfield(cont, field)
            v = cont.(field);
        else
            v = [];
        end
        v = v(:);  % force column
        if nargin < 2, fillval = NaN; end
        if numel(v) < n
            v(end+1:n,1) = fillval;
        elseif numel(v) > n
            v = v(1:n,1);
        end
    end

    % build columns (all n×1 guaranteed)
    OutagedBranch   = col('branch_idx', NaN);
    FromBus         = col('fbus', NaN);
    ToBus           = col('tbus', NaN);
    PF_Converged    = logical(col('pf_converged', 0));

    NumVunder       = col('num_vunder', NaN);
    NumVover        = col('num_vover', NaN);
    NumVoltageViol  = col('num_vviol',  -Inf);     % nonconv/missing -> severe
    WorstVBus       = col('worst_v_bus', NaN);
    WorstVM_pu      = col('worst_vm', NaN);
    VminLimit_pu    = col('worst_vmin', NaN);
    VmaxLimit_pu    = col('worst_vmax', NaN);
    WorstVDev_pu    = col('worst_vdev_pu', -Inf);  % nonconv/missing -> severe

    % assemble table
    T = table(OutagedBranch, FromBus, ToBus, PF_Converged, ...
              NumVunder, NumVover, NumVoltageViol, ...
              WorstVBus, WorstVM_pu, VminLimit_pu, VmaxLimit_pu, WorstVDev_pu);

    % sort: PF_Converged (false first), then NumVoltageViol desc, then WorstVDev desc
    sortMat = [double(T.PF_Converged), -T.NumVoltageViol, -T.WorstVDev_pu];
    [~, order] = sortrows(sortMat, [1 2 3]);
    resultsV = T(order,:);
end
% %%
% what happend in line 36 outage

% ---------- Contingency case ----------
mpcC = ac118;
mpcC.branch(br_out, BR_STATUS) = 0;


for i = 31
    br_out = meshed_branch_idx(i);

    mpc = ac118;                     % start from AC-initialized case
    mpc.branch(br_out, BR_STATUS) = 0;

    r = runpf(mpc, mpopt);

rAC = runpf(mpcC, mpopt);
rDC = rundcpf(mpcC, mpopt);
end

fprintf('\n================== Contingency Report: Outage Branch %d ==================\n', br_out);
fprintf('Outaged branch endpoints: %d -- %d\n', mpcC.branch(br_out,F_BUS), mpcC.branch(br_out,T_BUS));

if ~rAC.success
    fprintf('ACPF did NOT converge for this contingency.\n');
else
    fprintf('ACPF converged.\n');
end
if ~rDC.success
    fprintf('DCPF did NOT converge for this contingency.\n');
else
    fprintf('DCPF converged.\n');
end


% === AC branch apparent power (per end), then max end ===
Pac_from = rAC.branch(:, PF);
Qac_from = rAC.branch(:, QF);
Pac_to   = rAC.branch(:, PT);
Qac_to   = rAC.branch(:, QT);

Sac_fr = hypot(Pac_from, Qac_from);      % sqrt(P^2 + Q^2)
Sac_to = hypot(Pac_to,   Qac_to);
Sbrac  = max(Sac_fr, Sac_to);

RateA_ac = rAC.branch(:, RATE_A);     % should match mpc.branch(:,RATE_A)

overload_ac = Sbrac - RateA_ac;
acvio_idx = find(overload_ac > tolS);

% === DC branch "S" (Q is (usually) ~0), then max end ===
Pdc_from = rDC.branch(:, PF);
Qdc_from = rDC.branch(:, QF);         % typically 0 in DCOPF results
Pdc_to   = rDC.branch(:, PT);
Qdc_to   = rDC.branch(:, QT);

Sdc_fr = hypot(Pdc_from, Qdc_from);
Sdc_to = hypot(Pdc_to,   Qdc_to);
Sbrdc  = max(Sdc_fr, Sdc_to);

RateA_dc = rDC.branch(:, RATE_A);

overload_dc = Sbrdc - RateA_dc;
dcvio_idx = find(overload_dc > tolS);

%  FIGURE 1: Branch flows vs limits (grouped columns)
% -----------------------------
nbr = size(ac118.branch, 1);
br_idx = (1:nbr)';

% --- After you compute Sbrac, Sbrdc, and have ac_opf, dc_opf ---
RateA = ac118.branch(:, RATE_A);
br = (1:length(RateA)).';

figure; hold on; box on;

% Bar plot for RateA (thermal limits)
hb = bar(br, RateA, 1.0);  % 1.0 = full bar width
hb.FaceAlpha = 0.25;       % make it transparent
hb.EdgeColor = 'none';

% Line plots for branch apparent power
h1 = plot(br, Sbrac, 'LineWidth', 1.4);
h2 = plot(br, Sbrdc, 'LineWidth', 1.4);

grid on;
xlabel('Branch index');
ylabel('Apparent power |S| (MVA)');
title('Branch Apparent Power vs Thermal Limit');

% Optional: highlight only violations
tol = 1e-6;  % numerical tolerance (see note below)
ac_vio = find(Sbrac - RateA > tol);
dc_vio = find(Sbrdc - RateA > tol);

h3 = plot(ac_vio, Sbrac(ac_vio), 'o', 'MarkerSize', 6, 'LineWidth', 1.2);
h4 = plot(dc_vio, Sbrdc(dc_vio), 's', 'MarkerSize', 6, 'LineWidth', 1.2);

legend([h1 h2 hb h3 h4], ...
       {'Sbr (ACOPF)', 'Sbr (DCOPF)', 'RATE\_A', 'AC viol.', 'DC viol.'}, ...
       'Location', 'best');










% % Helper to compute |S| per branch
% calcS = @(br) max(hypot(br(:,PF),br(:,QF)), hypot(br(:,PT),br(:,QT)));
% 
% % ---------- Identify worst line after outage ----------
% if rAC.success
%     SAC  = calcS(rAC.branch);
%     Rate = rAC.branch(:, RATE_A); Rate(Rate<=0)=Inf;
%     in   = rAC.branch(:, BR_STATUS) > 0;
%     ratio = zeros(size(SAC)); ratio(in) = SAC(in)./Rate(in);
%     [wrAC, wlAC] = max(ratio);
%     fprintf('\n--- ACPF post-contingency worst line ---\n');
%     fprintf('WorstLine=%d, |S|=%.2f, Limit=%.2f, OverPct=%.4f%%\n', ...
%         wlAC, SAC(wlAC), Rate(wlAC), max(0,(wrAC-1)*100));
% end
% 
% if rDC.success
%     SDC  = calcS(rDC.branch);
%     Rate = rDC.branch(:, RATE_A); Rate(Rate<=0)=Inf;
%     in   = rDC.branch(:, BR_STATUS) > 0;
%     ratio = zeros(size(SDC)); ratio(in) = SDC(in)./Rate(in);
%     [wrDC, wlDC] = max(ratio);
%     fprintf('\n--- DCPF post-contingency worst line ---\n');
%     fprintf('WorstLine=%d, |S|=%.2f, Limit=%.2f, OverPct=%.4f%%\n', ...
%         wlDC, SDC(wlDC), Rate(wlDC), max(0,(wrDC-1)*100));
% end
% 
% % ---------- Specifically track line 31 if you care about it ----------
% line_of_interest = 31;
% 
% if rAC.success
%     SAC0 = calcS(rAC0.branch);
%     SAC1 = calcS(rAC.branch);
%     RateA = rAC.branch(:, RATE_A);
%     fprintf('\nACPF Line %d: base |S|=%.2f, post |S|=%.2f, limit=%.2f, margin(post-limit)=%.2f\n', ...
%         line_of_interest, SAC0(line_of_interest), SAC1(line_of_interest), ...
%         RateA(line_of_interest), SAC1(line_of_interest)-RateA(line_of_interest));
% end
% 
% if rDC.success
%     SDC0 = calcS(rDC0.branch);
%     SDC1 = calcS(rDC.branch);
%     RateD = rDC.branch(:, RATE_A);
%     fprintf('DCPF Line %d: base |S|=%.2f, post |S|=%.2f, limit=%.2f, margin(post-limit)=%.2f\n', ...
%         line_of_interest, SDC0(line_of_interest), SDC1(line_of_interest), ...
%         RateD(line_of_interest), SDC1(line_of_interest)-RateD(line_of_interest));
% end









% Pac_from = ac_opf.branch(:, PF);
% Qac_from = ac_opf.branch(:, QF);
% Pac_to   = ac_opf.branch(:, PT);
% Qac_to   = ac_opf.branch(:, QT);
% % Pac = max(abs(Pac_from), abs(Pac_to));
% % Qac = max(abs(Qac_from), abs(Qac_to));
% 
% Sac_fr = hypot(Pac_from, Qac_from);   % sqrt(P^2 + Q^2)
% Sac_to = hypot(Pac_to,   Qac_to);
% Sbrac  = max(Sac_fr, Sac_to);
% 
% 
% 
% Pdc_from = dc_opf.branch(:, PF);
% Qdc_from = dc_opf.branch(:, QF);
% Pdc_to   = dc_opf.branch(:, PT);
% Qdc_to   = dc_opf.branch(:, QT);
% % Pdc = max(abs(Pdc_from), abs(Pdc_to));
% % Qdc = max(abs(Qdc_from), abs(Qdc_to));
% 
% Sdc_fr = hypot(Pdc_from, Qdc_from);   % sqrt(P^2 + Q^2)
% Sdc_to = hypot(Pdc_to,   Qdc_to);
% Sbrdc  = max(Sdc_fr, Sdc_to);
% 
% overload_ac = Sbrac - ac_opf.branch(:, RATE_A);   % >0 means violation
% acvio_idx = find(overload_ac > 1e-6);
% 
% overload_dc = Sbrdc - dc_opf.branch(:, RATE_A);   % >0 means violation
% dcvio_idx = find(overload_dc > 1e-6);
% 
% 
% 
% %%
% 
% 
% PF = max (res_opf.branch(:, PF), res_opf.branch(:, PT));
% PT = max (res_opf.branch(:, QF), res_opf.branch(:, QT));
% 
% S_fr = sqrt(res_opf.branch(:, PF).^2 + res_opf.branch(:, QF).^2);
% S_to = sqrt(res_opf.branch(:, PT).^2 + res_opf.branch(:, QT).^2);
% Sbr  = max([S_fr, S_to], [], 2);
% % S_violation = max(0, Sbr - Smax);
% % 
% % S_violation
% 
%% contingency analysis using AC power flow
% minRate = min(mpc.branch(:, RATE_A));
% nZero   = sum(mpc.branch(:, RATE_A) <= 1e-6);
% fprintf('min RATE_A = %.6f, #near-zero limits = %d\n', minRate, nZero);% run optimal power flow and save the result
% res_opf = runopf(mpc);
% define the number of branches, indices of branches whose removal islands
% the system, and the indices of the rest of the branches
% 
% ibr_radial = [8,9,10,26,30,38,63,64,65,68,81];
% ibr_meshed = setdiff(1:nbr, ibr_radial);
% % ibr_meshed = [1];
% smax = zeros(length(ibr_meshed), 2);
% ct = 1;
% for ibr = ibr_meshed
%     ctg = res_opf;
%     ctg.branch(ibr, BR_STATUS) = 0;
%     res_con = runpf(ctg);
%     S_fr = sqrt(res_con.branch(:, PF).^2 + res_con.branch(:, QF).^2);
%     S_to = sqrt(res_con.branch(:, PT).^2 + res_con.branch(:, QT).^2);
%     Sbr = max([S_fr, S_to], [], 2);
%     smax(ct, 1) = max(Sbr - res_con.branch(:, RATE_A));
%     ct = ct + 1;
% end
% 
% smax
% 
% %% Security-Constrained DC OPF
% mpopt = mpoption('verbose', verbose);
% mpopt = mpoption(mpopt, 'out.gen', 1);
% mpopt = mpoption(mpopt, 'model', 'DC');
% mpopt = mpoption(mpopt, 'opf.dc.solver', 'MIPS');
% mpopt = mpoption(mpopt, 'most.solver', 'DEFAULT');
% if ~verbose
%     mpopt = mpoption(mpopt, 'out.all', 0);
% end
% 
% % contingency table
% % label probty  type        row column      chgtype newvalue
% contab = [
%   % 1   0    CT_TGEN     2    GEN_STATUS  CT_REP  0;      %% gen 2 at bus 2
%     1   0    CT_TBRCH    186   BR_STATUS   CT_REP  0;      %% line 36
% ];
% 
% mdi = loadmd(mpc, [], [], [], contab);
% mdo = most(mdi, mpopt);                  % solve SCOPF
% most_summary(mdo);
% 
% %% Check power flow feasibility
% feas = res_opf;
% feas.gen(:, PG) = mdo.results.ExpectedDispatch;
% vv = zeros(length(ibr_meshed),1); % check if there are voltage magnitude violations
% ct = 1;
% for ibr = ibr_meshed
%     ctg = feas;
%     ctg.branch(ibr, BR_STATUS) = 0;
%     res_con = runpf(ctg);
%     vv(ct) = sum(res_con.bus(:, VM) <= res_con.bus(:, VMIN) | ...
%         res_con.bus(:, VM) >= res_con.bus(:, VMAX));
%     S_fr = sqrt(res_con.branch(:, PF).^2 + res_con.branch(:, QF).^2);
%     S_to = sqrt(res_con.branch(:, PT).^2 + res_con.branch(:, QT).^2);
%     Sbr = max([S_fr, S_to], [], 2);
%     smax(ct, 2) = max(Sbr - res_con.branch(:, RATE_A));
%     ct = ct + 1;
% end
% 
% % comparison of worst line flow violations for each contingency before and
% % after SCOPF
% smax
