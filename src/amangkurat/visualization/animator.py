"""3D visualization for Klein-Gordon solitons."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 13


class Animator:
    """3D Klein-Gordon animations."""
    
    @staticmethod
    def create_gif(result: dict, filename: str, output_dir: str = "outputs",
                  title: str = "Klein-Gordon", fps: int = 30, dpi: int = 150,
                  colormap: str = 'plasma'):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        x = result['x']
        t = result['t']
        phi = result['phi']
        
        print(f"    Creating 3D animation ({len(t)} frames)...")
        
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('#000814')
        ax.set_facecolor('#000814')
        
        ax.set_xlabel('Position', fontsize=14, color='white', labelpad=15)
        ax.set_ylabel('Time', fontsize=14, color='white', labelpad=15)
        ax.set_zlabel('Field φ', fontsize=14, color='white', labelpad=15)
        ax.tick_params(colors='white', labelsize=11)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('cyan')
        ax.yaxis.pane.set_edgecolor('cyan')
        ax.zaxis.pane.set_edgecolor('cyan')
        ax.grid(True, alpha=0.15, color='cyan', linestyle='--', linewidth=0.5)
        
        cmap = plt.get_cmap(colormap)
        
        dummy_surf = ax.plot_surface(
            np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)),
            cmap=colormap, vmin=np.min(phi), vmax=np.max(phi)
        )
        cbar = fig.colorbar(dummy_surf, ax=ax, pad=0.12, shrink=0.6)
        cbar.set_label('Field φ', color='white', fontsize=16)
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=11)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        dummy_surf.remove()
        
        title_text = ax.text2D(
            0.5, 0.97, title, transform=ax.transAxes,
            fontsize=18, color='white', weight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#000814',
                     alpha=0.9, edgecolor='cyan', linewidth=2.5)
        )
        
        time_text = ax.text2D(
            0.5, 0.02, '', transform=ax.transAxes,
            fontsize=22, color='#00ff41', weight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='#000814',
                     alpha=0.95, edgecolor='#00ff41', linewidth=3)
        )
        
        def animate(frame):
            ax.clear()
            ax.set_facecolor('#000814')
            
            phi_subset = phi[:frame+1].T
            t_subset = t[:frame+1]
            T_sub, X_sub = np.meshgrid(t_subset, x)
            
            surf = ax.plot_surface(
                X_sub, T_sub, phi_subset,
                cmap=colormap, alpha=0.9,
                linewidth=0, antialiased=True,
                vmin=np.min(phi), vmax=np.max(phi),
                edgecolor='none', shade=True
            )
            
            ax.plot(x, [t[frame]]*len(x), phi[frame],
                   color='#00ff41', linewidth=4, alpha=1.0, zorder=10)
            
            ax.set_xlabel('Position', fontsize=14, color='white', labelpad=15)
            ax.set_ylabel('Time', fontsize=14, color='white', labelpad=15)
            ax.set_zlabel('Field φ', fontsize=14, color='white', labelpad=15)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(0, t[-1])
            ax.set_zlim(np.min(phi)*1.15, np.max(phi)*1.15)
            
            elev = 20 + 10*np.sin(frame*0.02)
            azim = 45 + frame*0.25
            ax.view_init(elev=elev, azim=azim)
            
            ax.tick_params(colors='white', labelsize=11)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('cyan')
            ax.yaxis.pane.set_edgecolor('cyan')
            ax.zaxis.pane.set_edgecolor('cyan')
            ax.grid(True, alpha=0.15, color='cyan', linestyle='--', linewidth=0.5)
            
            time_text.set_text(f't = {t[frame]:.3f}')
            
            return [surf, title_text, time_text]
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(t),
            interval=1000/fps, blit=False, repeat=True
        )
        
        print(f"    Rendering GIF...")
        writer = animation.PillowWriter(fps=fps)
        
        with tqdm(total=len(t), desc="    Progress", unit="frame") as pbar:
            def progress_callback(current_frame, total_frames):
                pbar.n = current_frame + 1
                pbar.refresh()
            
            anim.save(filepath, writer=writer, dpi=dpi,
                     progress_callback=progress_callback)
        
        plt.close(fig)
        print(f"    Animation saved: {filepath}")
