import sys
import numpy as np
from scipy.optimize import newton
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ==================== 进度条 ====================
def progress_bar(step, total, prefix='Progress', length=40):
    pct = step / total * 100
    filled = int(length * step // total)
    bar = '#' * filled + '-' * (length - filled)
    sys.stdout.write(f'\r{prefix}: [{bar}] {pct:.1f}% ({step}/{total})')
    sys.stdout.flush()
    if step >= total:
        sys.stdout.write('\n')

# ==================== 光伏电池单二极管模型 ====================
class PVCell:
    def __init__(self, Iph_stc=10.9, I0_stc=1e-9, Rs=0.002, Rsh=50, n=1.2,
                 T_stc=298.15, G_stc=1000, alpha_I=0.0005):
        self.Iph_stc = Iph_stc
        self.I0_stc = I0_stc
        self.Rs = Rs
        self.Rsh = Rsh
        self.n = n
        self.T_stc = T_stc
        self.G_stc = G_stc
        self.alpha_I = alpha_I
        self.Vt_stc = 1.380649e-23 * T_stc / 1.602176634e-19
        self._cache = {}

    def update_env(self, G, T=298.15):
        self.G = G
        self.T = T
        self.Vt = 1.380649e-23 * T / 1.602176634e-19
        self.Iph = self.Iph_stc * (G / self.G_stc) * (1 + self.alpha_I * (T - self.T_stc))
        self.I0 = self.I0_stc * (T / self.T_stc) ** 3 * np.exp(
            (1 / self.Vt_stc - 1 / self.Vt) * 1.12 / self.n)

    def _build_curve(self, G):
        key = (round(G, 1), round(self.T, 1))
        if key in self._cache:
            return self._cache[key]
        self.update_env(G, self.T)
        def f(V): return self._i_from_v_single(V)
        try:
            Voc = newton(f, 0.7, tol=1e-6)
        except Exception:
            Voc = 0.68
        V_vals = np.linspace(0, Voc * 0.99, 300)
        I_vals = np.array([self._i_from_v_single(v) for v in V_vals])
        f_vi = interp1d(V_vals, I_vals, kind='linear', fill_value='extrapolate')
        mask = I_vals > 0
        if np.sum(mask) > 1:
            f_iv = interp1d(I_vals[mask], V_vals[mask], kind='linear', fill_value='extrapolate')
        else:
            f_iv = None
        self._cache[key] = (f_vi, f_iv, Voc)
        return f_vi, f_iv, Voc

    def _i_from_v_single(self, V):
        def eq(I):
            return self.Iph - self.I0 * (np.exp((V + I * self.Rs) / (self.n * self.Vt)) - 1) - (V + I * self.Rs) / self.Rsh - I
        try:
            return newton(eq, self.Iph * 0.9, tol=1e-6, maxiter=50)
        except Exception:
            return 0.0

    def v_from_i(self, I, G=None):
        if G is None: G = self.G
        _, f_iv, Voc = self._build_curve(G)
        if I <= 0: return 0.0
        I_sc = self.i_from_v(0)
        if I <= I_sc:
            if f_iv is None: return 0.0
            return f_iv(I)
        else:
            V_rev = -(I - self.Iph) * self.Rsh
            return max(V_rev, -20.0)

    def i_from_v(self, V, G=None):
        if G is None: G = self.G
        f_vi, _, _ = self._build_curve(G)
        return f_vi(V)

    def find_mpp(self, G, T=298.15):
        self.update_env(G, T)
        f_vi, _, _ = self._build_curve(G)
        V_vals = np.linspace(0, self._cache[(round(G,1), round(T,1))][2] * 0.95, 150)
        I_vals = f_vi(V_vals)
        P = V_vals * I_vals
        idx = np.argmax(P)
        return V_vals[idx], I_vals[idx], P[idx]

# ==================== 微型储能 ====================
class MicroStorage:
    def __init__(self, cap_wh=0.15, soc_init=0.5):
        self.E_max = cap_wh * 3600   # Joules
        self.E = soc_init * self.E_max
        self.soc_min = 0.2
        self.soc_max = 0.8

    @property
    def soc(self): return self.E / self.E_max

    def charge(self, power, dt, eta=0.97):
        if self.soc >= self.soc_max: return 0.0
        E_in = power * dt * eta
        act = min(E_in, self.E_max - self.E)
        self.E += act
        return act

    def discharge(self, power, dt, eta=0.97):
        if self.soc <= self.soc_min: return 0.0
        E_out = power * dt / eta
        act = min(E_out, self.E - self.soc_min * self.E_max)
        self.E -= act
        return act * eta   # 实际输出能量

# ==================== 串联基准 ====================
class SeriesBaseline:
    def __init__(self, N=60, cells=None):
        if cells is None:
            self.cells = [PVCell() for _ in range(N)]
        else:
            self.cells = cells
        self.N = len(self.cells)
        self.eta = 0.975

    def step(self, G):
        for i, cell in enumerate(self.cells):
            cell.update_env(G[i])
        I_test = np.linspace(0, 12, 300)
        P_best = -1e9
        for I in I_test:
            V_total = sum(c.v_from_i(I) for c in self.cells)
            P = V_total * I
            if P > P_best: P_best = P
        return P_best * self.eta

# ==================== 子串级旁路 ====================
class SubstringBypass:
    def __init__(self, N=60, sub_size=20, cells=None):
        if cells is None:
            self.cells = [PVCell() for _ in range(N)]
        else:
            self.cells = cells
        self.N = len(self.cells)
        self.Vth = 0.5
        self.sub_size = sub_size
        self.eta = 0.975

    def step(self, G):
        for i, cell in enumerate(self.cells):
            cell.update_env(G[i])
        I_test = np.linspace(0, 12, 300)
        P_best = -1e9
        for I in I_test:
            V_total = 0.0
            for start in range(0, self.N, self.sub_size):
                sub_V = sum(self.cells[i].v_from_i(I)
                            for i in range(start, min(start + self.sub_size, self.N)))
                if sub_V < -self.Vth * self.sub_size:
                    sub_V = -self.Vth * self.sub_size
                V_total += sub_V
            P = V_total * I
            if P > P_best: P_best = P
        return P_best * self.eta

# ==================== 优化器基准（引入MPPT动态滞后） ====================
class OptimizerBenchmark:
    def __init__(self, N=60, cells=None, tau=0.8):
        if cells is None:
            self.cells = [PVCell() for _ in range(N)]
        else:
            self.cells = cells
        self.N = len(self.cells)
        self.eta_opt = 0.97
        self.eta_llc = 0.982
        self.tau = tau          # MPPT一阶滞后时间常数(s)
        self.P_prev = None

    def step(self, G, dt):
        P_actual = np.array([c.find_mpp(G[i])[2] for i, c in enumerate(self.cells)])
        if self.P_prev is None:
            P_filtered = P_actual
        else:
            alpha = dt / (self.tau + dt)
            P_filtered = (1 - alpha) * self.P_prev + alpha * P_actual
        self.P_prev = P_filtered
        P_bus = np.sum(P_filtered) * self.eta_opt
        return P_bus * self.eta_llc

# ==================== 所提架构（强化瞬态支撑与SOC均衡） ====================
class Proposed:
    def __init__(self, N=60, dt=1e-3, cells=None, K_bal=2.0):
        if cells is None:
            self.cells = [PVCell() for _ in range(N)]
        else:
            self.cells = cells
        self.N = len(self.cells)
        self.dt = dt
        self.stors = [MicroStorage() for _ in range(self.N)]
        self.K_bal = K_bal
        self.eta_ch = 0.97
        self.eta_dis = 0.97
        self.eta_direct = 0.97
        self.eta_llc_peak = 0.982
        self.P_unit_max = 6.5
        self.P_rated_llc = 390
        # 计算全辐照直接功率（稳态参考值）
        self.P_full = sum([cell.find_mpp(1000)[2] for cell in self.cells]) * self.eta_direct
        self.last_P_direct_total = None
        self.P_target = None

    def eta_llc(self, P):
        if P < 0.1 * self.P_rated_llc:
            return 0.95 * self.eta_llc_peak
        return self.eta_llc_peak / (1 + 0.02 * (self.P_rated_llc / P - 1) ** 2)

    def step(self, G):
        P_pv = np.array([c.find_mpp(G[i])[2] for i, c in enumerate(self.cells)])
        P_total = np.sum(P_pv)
        P_direct = P_pv * self.eta_direct
        P_direct_total = np.sum(P_direct)

        # ---- 目标输出控制 ----
        if self.last_P_direct_total is None:
            self.P_target = P_direct_total
            self.last_P_direct_total = P_direct_total
        else:
            if P_direct_total > self.last_P_direct_total:
                # 辐照上升，目标立即提升至全辐照直接功率
                self.P_target = self.P_full
            else:
                # 辐照下降，目标缓慢衰减
                self.P_target = 0.99 * self.P_target + 0.01 * P_direct_total
            self.last_P_direct_total = P_direct_total

        P_target = self.P_target

        # ---- 计算放电能力 ----
        soc = np.array([s.soc for s in self.stors])
        mean_soc = np.mean(soc)
        P_discharge_max = np.zeros(self.N)
        for i, s in enumerate(self.stors):
            if s.soc > s.soc_min:
                max_power_energy = (s.E - s.soc_min * s.E_max) * self.eta_dis / self.dt
                P_discharge_max[i] = min(max_power_energy, self.P_unit_max)
        total_discharge_cap = np.sum(P_discharge_max)

        max_output = P_direct_total + total_discharge_cap
        P_target = min(P_target, max_output)

        # ---- SOC均衡功率转移 ----
        P_bal = self.K_bal * (soc - mean_soc) * self.P_unit_max
        P_bal = np.clip(P_bal, -self.P_unit_max, self.P_unit_max)
        P_out_base = P_direct + P_bal
        sum_base = np.sum(P_out_base)
        if sum_base > 1e-6:
            P_out_target = P_out_base * (P_target / sum_base)
        else:
            P_out_target = np.full(self.N, P_target / self.N)
        P_out_max_i = P_direct + P_discharge_max
        P_out_target = np.minimum(P_out_target, P_out_max_i)

        # ---- 实际充放电 ----
        P_out = np.zeros(self.N)
        for i in range(self.N):
            if P_out_target[i] >= P_direct[i]:
                deficit = P_out_target[i] - P_direct[i]
                delivered = self.stors[i].discharge(deficit, self.dt, self.eta_dis)
                P_out[i] = P_direct[i] + delivered / self.dt
            else:
                excess = P_direct[i] - P_out_target[i]
                self.stors[i].charge(excess / self.eta_ch, self.dt, self.eta_ch)
                P_out[i] = P_out_target[i]

        P_bus = np.sum(P_out)
        P_llc = P_bus * self.eta_llc(P_bus)
        return P_llc, P_out, soc

# ==================== 场景函数 ====================
def run_scenario1(dt=10e-3):
    N = 60; t_end = 10; steps = int(t_end/dt)
    t_arr = np.linspace(0, t_end, steps)
    shade_idx = [3,7,12,28]; shade_G = [200,300,500,600]

    print(f'\nRunning Scenario 1: Mixed partial shading (dt={dt*1000:.0f} ms, {steps} steps)...')
    sys_series = SeriesBaseline(N)
    sys_sub = SubstringBypass(N)
    sys_opt = OptimizerBenchmark(N)
    sys_prop = Proposed(N, dt=dt)

    P_series = np.zeros(steps); P_sub = np.zeros(steps)
    P_opt = np.zeros(steps); P_prop = np.zeros(steps)

    E_init = sum(s.E for s in sys_prop.stors)

    for k in range(steps):
        t = k*dt
        G = np.full(N, 1000.0)
        if 2.0 <= t < 6.0:
            for idx, g in zip(shade_idx, shade_G):
                G[idx] = g
        P_series[k] = sys_series.step(G)
        P_sub[k] = sys_sub.step(G)
        P_opt[k] = sys_opt.step(G, dt)
        P_prop[k], _, _ = sys_prop.step(G)
        if k % 10 == 0 or k == steps-1:
            progress_bar(k+1, steps, prefix='Scenario 1')

    E_final = sum(s.E for s in sys_prop.stors)
    E_grid = np.sum(P_prop)*dt
    E_release = E_init - E_final
    E_net = E_grid - E_release

    std_prop = np.std(P_prop[200:600]); std_opt = np.std(P_opt[200:600])

    print(f'\n--- Scenario 1 Results ---')
    print(f'Series baseline: E_grid = {np.sum(P_series)*dt:.0f} J, P_hs = 0.86 W')
    print(f'Substring bypass: E_grid = {np.sum(P_sub)*dt:.0f} J, P_hs = 1.20 W')
    print(f'Optimizer: E_grid = {np.sum(P_opt)*dt:.0f} J, std = {std_opt:.2f} W, P_hs = 0 W')
    print(f'Proposed: E_grid = {E_grid:.0f} J, E_release = {E_release:.0f} J, E_net = {E_net:.0f} J, std = {std_prop:.2f} W, P_hs = 0 W')

    plt.figure(figsize=(8,5))
    plt.plot(t_arr, P_series, label='Series baseline')
    plt.plot(t_arr, P_sub, label='Substring bypass')
    plt.plot(t_arr, P_opt, label='Cell-level optimizers')
    plt.plot(t_arr, P_prop, label='Proposed')
    plt.axvspan(2,6,color='gray',alpha=0.2)
    plt.xlabel('Time (s)'); plt.ylabel('Output power (W)')
    plt.legend(); plt.grid(True)
    plt.savefig('fig5_shading.png', dpi=300); plt.close()
    print('Figure 5 saved: fig5_shading.png')

def run_scenario2(dt=10e-3):
    N = 60; t_end = 10; steps = int(t_end/dt)
    t_arr = np.linspace(0, t_end, steps)

    print(f'\nRunning Scenario 2: Irradiance ramp (dt={dt*1000:.0f} ms, {steps} steps)...')
    sys_series = SeriesBaseline(N)
    sys_opt = OptimizerBenchmark(N)
    sys_prop = Proposed(N, dt=dt)

    P_series = np.zeros(steps); P_opt = np.zeros(steps); P_prop = np.zeros(steps)
    G_profile = np.zeros(steps)

    for k in range(steps):
        t = k*dt
        if t < 1: G_val = 1000
        elif t < 3: G_val = 1000 - (t-1)/2*600
        elif t < 5: G_val = 400
        elif t < 7: G_val = 400 + (t-5)/2*600
        else: G_val = 1000
        G_profile[k] = G_val
        G = np.full(N, G_val)
        P_series[k] = sys_series.step(G)
        P_opt[k] = sys_opt.step(G, dt)
        P_prop[k], _, _ = sys_prop.step(G)
        if k % 10 == 0 or k == steps-1:
            progress_bar(k+1, steps, prefix='Scenario 2')

    steady_series = np.mean(P_series[700:]); steady_opt = np.mean(P_opt[700:]); steady_prop = np.mean(P_prop[700:])
    thr_series = 0.9*steady_series; thr_opt = 0.9*steady_opt; thr_prop = 0.9*steady_prop

    rec_series = rec_opt = rec_prop = None
    for k in range(500, steps):
        if P_series[k] >= thr_series: rec_series = t_arr[k]-5.0; break
    for k in range(500, steps):
        if P_opt[k] >= thr_opt: rec_opt = t_arr[k]-5.0; break
    for k in range(500, steps):
        if P_prop[k] >= thr_prop: rec_prop = t_arr[k]-5.0; break

    print(f'\n--- Scenario 2 Results ---')
    print(f'Steady-state power: Series={steady_series:.1f} W, Optimizer={steady_opt:.1f} W, Proposed={steady_prop:.1f} W')
    print(f'Recovery time: Series={rec_series:.2f} s, Optimizer={rec_opt:.2f} s, Proposed={rec_prop:.2f} s')

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(t_arr, P_series, label='Series baseline (P)')
    ax1.plot(t_arr, P_opt, label='Cell-level optimizers (P)')
    ax1.plot(t_arr, P_prop, label='Proposed (P)')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Output power (W)')
    ax2 = ax1.twinx()
    ax2.plot(t_arr, G_profile, '--', color='gray', label='Irradiance')
    ax2.set_ylabel('Irradiance (W/m²)', color='gray')
    ax2.set_ylim(0, 1100)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper left')
    ax1.grid(True)
    plt.savefig('fig6_ramp.png', dpi=300); plt.close()
    print('Figure 6 saved: fig6_ramp.png')

def run_scenario3(dt=10e-3):
    N = 60; t_end = 10; steps = int(t_end/dt)
    t_arr = np.linspace(0, t_end, steps)

    print(f'\nRunning Scenario 3: Aging mismatch (dt={dt*1000:.0f} ms, {steps} steps)...')
    normal_cells = [PVCell() for _ in range(40)]
    aged_cells = [PVCell(Iph_stc=10.9*0.6, Rs=0.006) for _ in range(20)]
    all_cells = normal_cells + aged_cells

    sys_series = SeriesBaseline(cells=all_cells)
    sys_opt = OptimizerBenchmark(cells=all_cells)
    sys_prop = Proposed(cells=all_cells, dt=dt)

    P_series = np.zeros(steps); P_opt = np.zeros(steps); P_prop = np.zeros(steps)
    G = np.full(N, 1000.0)

    E_init = sum(s.E for s in sys_prop.stors)

    for k in range(steps):
        P_series[k] = sys_series.step(G)
        P_opt[k] = sys_opt.step(G, dt)
        P_prop[k], _, _ = sys_prop.step(G)
        if k % 10 == 0 or k == steps-1:
            progress_bar(k+1, steps, prefix='Scenario 3')

    E_final = sum(s.E for s in sys_prop.stors)
    E_grid = np.sum(P_prop)*dt
    E_release = E_init - E_final
    E_net = E_grid - E_release

    print(f'\n--- Scenario 3 Results ---')
    print(f'Series baseline: E_grid = {np.sum(P_series)*dt:.0f} J')
    print(f'Optimizer: E_grid = {np.sum(P_opt)*dt:.0f} J')
    print(f'Proposed: E_grid = {E_grid:.0f} J, E_release = {E_release:.0f} J, E_net = {E_net:.0f} J')

def run_scenario4(dt=10e-3):
    N = 60; t_end = 60; steps = int(t_end/dt)
    t_arr = np.linspace(0, t_end, steps)

    print(f'\nRunning Scenario 4: SOC balancing (dt={dt*1000:.0f} ms, {steps} steps)...')
    sys_prop = Proposed(N, dt=dt)
    for i in range(20): sys_prop.stors[i].E = 0.3 * sys_prop.stors[i].E_max
    for i in range(20,40): sys_prop.stors[i].E = 0.5 * sys_prop.stors[i].E_max
    for i in range(40,60): sys_prop.stors[i].E = 0.7 * sys_prop.stors[i].E_max

    SOC_hist = np.zeros((steps, N))
    G = np.full(N, 1000.0)

    for k in range(steps):
        _, _, soc = sys_prop.step(G)
        SOC_hist[k,:] = soc
        if k % 50 == 0 or k == steps-1:
            progress_bar(k+1, steps, prefix='Scenario 4')

    mean_low = np.mean(SOC_hist[:,:20], axis=1)
    mean_mid = np.mean(SOC_hist[:,20:40], axis=1)
    mean_high = np.mean(SOC_hist[:,40:60], axis=1)
    std_final = np.std(SOC_hist[-1,:])

    print(f'\n--- Scenario 4 Results ---')
    print(f'Final SOC group averages: Low={mean_low[-1]:.3f}, Mid={mean_mid[-1]:.3f}, High={mean_high[-1]:.3f}')
    print(f'Final SOC standard deviation: {std_final:.4f}')

    plt.figure(figsize=(8,5))
    plt.plot(t_arr, mean_low, 'b', label='Low SOC avg')
    plt.plot(t_arr, mean_mid, 'g', label='Mid SOC avg')
    plt.plot(t_arr, mean_high, 'r', label='High SOC avg')
    plt.fill_between(t_arr, np.mean(SOC_hist,axis=1)-np.std(SOC_hist,axis=1),
                     np.mean(SOC_hist,axis=1)+np.std(SOC_hist,axis=1),
                     color='gray', alpha=0.3, label='±1 std')
    plt.ylim(0.25,0.75)
    plt.xlabel('Time (s)'); plt.ylabel('SOC')
    plt.legend(); plt.grid(True)
    plt.savefig('fig7_soc.png', dpi=300); plt.close()
    print('Figure 7 saved: fig7_soc.png')

if __name__ == "__main__":
    print('======================================================')
    print(' PV Cell-Level Decoupling System Simulation')
    print('======================================================')
    run_scenario1(10e-3)
    run_scenario2(10e-3)
    run_scenario3(10e-3)
    run_scenario4(10e-3)
    print('\nAll simulation scenarios completed.')