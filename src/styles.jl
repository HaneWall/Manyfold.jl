using CairoMakie


function plot3D_data(X::AbstractMatrix{T}, X_label::AbstractVector{S}) where {S<:Real,T<:Real}
  CairoMakie.activate!(type="png")
  fig = Figure(size=(400, 400))
  ax = Axis3(fig[1, 1])
  scatter!(ax, X, color=X_label, colormap=(:Spectral, 0.6))
  return fig
end


function plot3D_data(X::AbstractMatrix{T}, X_oos::AbstractMatrix{T}, X_label::AbstractVector{S}, X_label_oos::AbstractVector{S}) where {S<:Real,T<:Real}
  CairoMakie.activate!(type="png")
  fig = Figure(size=(400, 400))
  ax = Axis3(fig[1, 1])
  scatter!(ax, X, color=X_label, colormap=(:Spectral, 0.6))
  scatter!(ax, X_oos, color=X_label_oos, colormap=(:Spectral, 0.6), strokewidth=1, strokecolor=:black)
  return fig
end


function plot3D_data(X::AbstractMatrix{T}, X_oos::AbstractMatrix{T}; backend=:CairoMakie) where {T<:Real}
  if backend == :CairoMakie
    CairoMakie.activate!(type="png")
  else
    GLMakie.activate!()
  end
  fig = Figure(size=(400, 400))
  ax = Axis3(fig[1, 1])
  scatter!(ax, X_oos, color=(:black, 0.1), strokewidth=1, strokecolor=:black)
  scatter!(ax, X, color=(:crimson, 0.4))
  return fig
end


function plot_matrix_coords(Proj::AbstractMatrix{T}, idx_x::Integer, X_label::AbstractVector{S}) where {S<:Real,T<:Real}
  CairoMakie.activate!(type="png")
  fig = Figure(
    size=(508, 608)
  )
  no_coords = size(Proj)[2]
  col = 1
  row = 1
  ax_list = Makie.Axis[]

  for idx_y in 1:no_coords
    ax = Axis(fig[row, col], xlabel="ψ_$(idx_x)", ylabel="ψ_$(idx_y)")
    push!(ax_list, ax)
    scatter!(ax_list[end], Proj[:, idx_x], Proj[:, idx_y], color=X_label, colormap=(:Spectral, 0.6))
    col += 1
    if col > 2
      col = 1
      row += 1
    end
  end
  return fig
end

function plot_matrix_coords(Proj::AbstractMatrix{T}, Proj_oos::AbstractMatrix{T}, idx_x::Integer, X_label::AbstractVector{S}, X_label_oos::AbstractVector{S}) where {S<:Real,T<:Real}
  CairoMakie.activate!(type="png")
  fig = Figure(
    size=(508, 608)
  )
  no_coords = size(Proj)[2]
  col = 1
  row = 1
  ax_list = Makie.Axis[]

  for idx_y in 1:no_coords
    ax = Axis(fig[row, col], xlabel="ψ_$(idx_x)", ylabel="ψ_$(idx_y)")
    push!(ax_list, ax)
    scatter!(ax_list[end], Proj[:, idx_x], Proj[:, idx_y], color=X_label, colormap=(:Spectral, 0.6))
    scatter!(ax_list[end], Proj_oos[:, idx_x], Proj_oos[:, idx_y], color=X_label_oos, colormap=(:Spectral, 0.6), strokecolor=:black, strokewidth=1)
    col += 1
    if col > 2
      col = 1
      row += 1
    end
  end
  return fig
end




function save_figs(arr; name="tmp")
  for (idx, fig) in enumerate(arr)
    save("./Figures/" * "$(name)" * "_$(idx).pdf", fig, pt_per_unit=1)
  end
end


export plot3D_data, plot_matrix_coords, save_figs
