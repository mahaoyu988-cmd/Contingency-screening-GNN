define_constants;
mpopt = mpoption('verbose', 0, 'out.all', 0);

%% ============================================================
%  24-hour IEEE 118 simulation with load + renewable curves
%  - bus 10 replaced by wind
%  - bus 26 replaced by solar PV
%  - renewables modeled as PV buses
%  - allowable PF = 0.9
%% ============================================================

%% -----------------------------
% Load base case
%% -----------------------------
mpc0 = loadcase('case118');

%% -----------------------------
% Save original bus loads
%% -----------------------------
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
        Smax(l) = 180;   % assume 138 kV
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

%% find generator rows
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

% Installed renewable capacity = replaced conventional PMAX
Pinst_wind  = Pmax10_old;
Pinst_solar = Pmax26_old;

% Q capability based on installed capacity
Qcap_wind  = Pinst_wind  * tanphi;
Qcap_solar = Pinst_solar * tanphi;

%% -----------------------------
% Turn off extra generators at those buses
%% -----------------------------
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

%% -----------------------------
% Make buses PV
%% -----------------------------
mpc0.bus(wind_bus,  BUS_TYPE) = PV;
mpc0.bus(solar_bus, BUS_TYPE) = PV;

%% -----------------------------
% Renewable cost settings for OPF
% Keep renewable marginal cost low
%% -----------------------------
if isfield(mpc0, 'gencost') && size(mpc0.gencost,1) >= size(mpc0.gen,1)
    for g = [keep10, keep26]
        mpc0.gencost(g, MODEL)    = 2;
        mpc0.gencost(g, STARTUP)  = 0;
        mpc0.gencost(g, SHUTDOWN) = 0;
        mpc0.gencost(g, NCOST)    = 3;
        mpc0.gencost(g, COST)     = 0;      % c2
        mpc0.gencost(g, COST+1)   = 0.01;   % c1
        mpc0.gencost(g, COST+2)   = 0;      % c0
    end
end

%% ============================================================
% 24-hour load and renewable curves
% You can replace these with your own profiles later
%% ============================================================

hours = (1:24).';

% Typical daily load multiplier (system-wide)
load_curve = [ ...
    0.62 0.58 0.55 0.54 0.56 0.62 0.72 0.82 0.90 0.95 0.98 1.00 ...
    0.99 0.98 1.00 1.05 1.10 1.18 1.22 1.20 1.10 0.95 0.80 0.68 ]'*1.325;

c = mean(load_curve);   % or choose c = 0.9
k = 1.3;                % try 1.2, 1.3, 1.4, etc.

load_curve_new = c + k * (load_curve - c);

% optional clipping
% xmin = 0.50;
% xmax = 1.35;
% load_curve_new = max(xmin, min(xmax, load_curve_new));

disp(load_curve_new)


% new_min = 0.5;
% new_max = 1.75;
% 
% load_curve = new_min + ...
%     (load_curve - min(load_curve)) / (max(load_curve) - min(load_curve)) ...
%     * (new_max - new_min);

% Typical wind availability curve
wind_curve = [ ...
    0.55 0.58 0.60 0.62 0.65 0.60 0.52 0.45 0.38 0.32 0.30 0.28 ...
    0.30 0.35 0.40 0.48 0.55 0.60 0.63 0.65 0.62 0.60 0.58 0.56 ]';

% Typical solar availability curve
solar_curve = [ ...
    0.00 0.00 0.00 0.00 0.00 0.02 0.06 0.14 0.28 0.46 0.66 0.83 ...
    0.94 1.00 0.97 0.88 0.70 0.48 0.22 0.06 0.00 0.00 0.00 0.00 ]';



%% ============================================================
% Storage arrays
%% ============================================================

nb = size(mpc0.bus, 1);
ng = size(mpc0.gen, 1);

success_opf = false(24,1);

total_load_MW   = zeros(24,1);
total_wind_MW   = zeros(24,1);
total_solar_MW  = zeros(24,1);
total_gen_MW    = zeros(24,1);

bus_vm_24       = zeros(nb,24);
bus_va_24       = zeros(nb,24);

gen_pg_24       = zeros(ng,24);
gen_qg_24       = zeros(ng,24);

branch_S_24     = zeros(nbr,24);
branch_viol_24  = zeros(24,1);

results24 = cell(24,1);

%% ============================================================
% Hourly loop
%% ============================================================

for h = 1:24
    mpc = mpc0;

    %% 1) Scale system load
    alpha = load_curve_new(h);
    mpc.bus(:, PD) = alpha * PD0;
    mpc.bus(:, QD) = alpha * QD0;

    %% 2) Renewable available power
    Pavail_wind  = wind_curve(h)  * Pinst_wind;
    Pavail_solar = solar_curve(h) * Pinst_solar;

    %% 3) Replace generator at bus 10 with wind
    mpc.gen(keep10, PG)         = Pavail_wind;
    mpc.gen(keep10, QG)         = 0;
    mpc.gen(keep10, QMAX)       =  Qcap_wind;
    mpc.gen(keep10, QMIN)       = -Qcap_wind;
    mpc.gen(keep10, VG)         = Vset_wind;
    mpc.gen(keep10, PMAX)       = Pavail_wind;   % availability limit
    mpc.gen(keep10, PMIN)       = 0;             % curtailment allowed
    mpc.gen(keep10, GEN_STATUS) = 1;

    %% 4) Replace generator at bus 26 with solar
    mpc.gen(keep26, PG)         = Pavail_solar;
    mpc.gen(keep26, QG)         = 0;
    mpc.gen(keep26, QMAX)       =  Qcap_solar;
    mpc.gen(keep26, QMIN)       = -Qcap_solar;
    mpc.gen(keep26, VG)         = Vset_solar;
    mpc.gen(keep26, PMAX)       = Pavail_solar;  % availability limit
    mpc.gen(keep26, PMIN)       = 0;             % curtailment allowed
    mpc.gen(keep26, GEN_STATUS) = 1;

    %% 5) Run ACOPF
    r = runopf(mpc, mpopt);

    if ~r.success
        fprintf('Hour %2d: ACOPF did not converge.\n', h);
        success_opf(h) = false;
        results24{h} = [];
        continue;
    end

    success_opf(h) = true;
    results24{h}   = r;

    %% 6) Store outputs
    total_load_MW(h)  = sum(r.bus(:, PD));
    total_wind_MW(h)  = r.gen(keep10, PG);
    total_solar_MW(h) = r.gen(keep26, PG);
    total_gen_MW(h)   = sum(r.gen(:, PG));

    gen_pg_24(:,h) = r.gen(:, PG);
    gen_qg_24(:,h) = r.gen(:, QG);

    Sf = hypot(r.branch(:, PF), r.branch(:, QF));
    St = hypot(r.branch(:, PT), r.branch(:, QT));
    Sbr = max(Sf, St);

    branch_S_24(:,h) = Sbr;

    rateA = r.branch(:, RATE_A);
    in    = r.branch(:, BR_STATUS) > 0;
    branch_viol_24(h) = sum(in & (Sbr > rateA + 1e-6));

    volt_viol_24   = zeros(24,1);
    num_vunder_24  = zeros(24,1);
    num_vover_24   = zeros(24,1);
    worst_vm_24    = zeros(24,1);
    worst_vbus_24  = zeros(24,1);
    worst_vdev_24  = zeros(24,1);

    %% 7) Voltage violations
    vm   = r.bus(:, VM);
    vmin = r.bus(:, VMIN);
    vmax = r.bus(:, VMAX);

    under = vm < (vmin - 1e-9);
    over  = vm > (vmax + 1e-9);
    viol  = under | over;

    num_vunder_24(h) = sum(under);
    num_vover_24(h)  = sum(over);
    volt_viol_24(h)  = sum(viol);

    if any(viol)
        dev = max(max(vmin - vm, 0), max(vm - vmax, 0));
        [dmax, bidx] = max(dev);

        worst_vm_24(h)   = vm(bidx);
        worst_vbus_24(h) = r.bus(bidx, BUS_I);
        worst_vdev_24(h) = dmax;
    else
        % if no violation, still record the minimum voltage bus
        [worst_vm_24(h), bidx] = min(vm);
        worst_vbus_24(h) = r.bus(bidx, BUS_I);
        worst_vdev_24(h) = 0;
    end

 



   fprintf(['Hour %2d | Load = %8.2f MW | Wind = %7.2f MW | Solar = %7.2f MW | ' ...
         'Branch Viol = %2d | Volt Viol = %2d (Under = %2d, Over = %2d) | ' ...
         'Worst VM = %.4f pu at bus %d\n'], ...
         h, total_load_MW(h), total_wind_MW(h), total_solar_MW(h), ...
         branch_viol_24(h), volt_viol_24(h), num_vunder_24(h), num_vover_24(h), ...
         worst_vm_24(h), worst_vbus_24(h));
end

%% ============================================================
% Summary table
%% ============================================================

HourlySummary = table( ...
    hours, load_curve_new, wind_curve, solar_curve, ...
    total_load_MW, total_wind_MW, total_solar_MW, total_gen_MW, ...
    branch_viol_24, volt_viol_24, num_vunder_24, num_vover_24, ...
    worst_vm_24, worst_vbus_24, worst_vdev_24, success_opf, ...
    'VariableNames', { ...
    'Hour','LoadMultiplier','WindCF','SolarCF', ...
    'TotalLoad_MW','WindPG_MW','SolarPG_MW','TotalGen_MW', ...
    'NumBranchViolations','NumVoltageViolations','NumVunder','NumVover', ...
    'WorstVM_pu','WorstVBus','WorstVDev_pu','OPF_Success'});

disp(HourlySummary);

%% ============================================================
% Plot 1: load and renewable curves
%% ============================================================

figure;
plot(hours, load_curve, 'x', 'LineWidth', 1.5); hold on;
plot(hours, load_curve_new, '-o', 'LineWidth', 1.5); hold on;
plot(hours, wind_curve, '-s', 'LineWidth', 1.5);
plot(hours, solar_curve, '-^', 'LineWidth', 1.5);
grid on;
xlabel('Hour');
ylabel('Per-unit multiplier / capacity factor');
title('24-hour load and renewable profiles');
legend('Typical Load curve','Load curve','Wind curve','Solar curve','Location','best');

%% ============================================================
% Plot 2: actual hourly MW
%% ============================================================

figure;
plot(hours, total_load_MW, '-o', 'LineWidth', 1.5); hold on;
plot(hours, total_wind_MW, '-s', 'LineWidth', 1.5);
plot(hours, total_solar_MW, '-^', 'LineWidth', 1.5);
grid on;
xlabel('Hour');
ylabel('MW');
title('24-hour load and renewable dispatch');
legend('Total load','Wind at bus 10','Solar at bus 26','Location','best');

%% ============================================================
% Plot 3: branch violation count by hour
%% ============================================================

figure;
plot(hours, branch_viol_24, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Hour');
ylabel('Number of branch thermal violations');
title('Hourly branch thermal violations');

%% ============================================================
% Plot 4: voltage violation count by hour
%% ============================================================

figure;
plot(hours, volt_viol_24, '-o', 'LineWidth', 1.5); hold on;
plot(hours, num_vunder_24, '-s', 'LineWidth', 1.5);
plot(hours, num_vover_24, '-^', 'LineWidth', 1.5);
grid on;
xlabel('Hour');
ylabel('Number of voltage violations');
title('Hourly voltage violations');
legend('Total voltage violations','Undervoltage','Overvoltage','Location','best');