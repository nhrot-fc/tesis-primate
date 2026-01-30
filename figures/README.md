# Figures Directory

Place your figures and images here. Supported formats include:
- PDF (recommended for vector graphics)
- PNG (for raster images)
- JPEG/JPG (for photographs)
- EPS (for older LaTeX workflows)

## Example Usage

In your LaTeX document, reference figures like this:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/your-image.pdf}
    \caption{Description of your figure}
    \label{fig:your-label}
\end{figure}
```

Then reference it in text with: `Figure~\ref{fig:your-label} shows...`
