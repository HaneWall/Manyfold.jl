using CairoMakie


function plot3D_data(X::AbstractMatrix{T}, X_label::AbstractVector{S}) where {S<:Real,T<:Real}
  CairoMakie.activate!(type="svg")
  fig = Figure(size=(400, 400))
  ax = Axis3(fig[1, 1])
  scatter!(ax, X, color=X_label, colormap=(:batlow, 0.6))
  return fig
end

function plot_matrix_coords(Proj::AbstractMatrix{T}, idx_x::Integer, X_label::AbstractVector{S}) where {S<:Real,T<:Real}
  CairoMakie.activate!(type="svg")
  fig = Figure(
    size=(508, 608)
  )
  no_coords = size(Proj)[1]
  col = 1
  row = 1
  ax_list = Makie.Axis[]

  for idx_y in 1:no_coords
    ax = Axis(fig[row, col])
    push!(ax_list, ax)
    scatter!(ax_list[end], Proj[idx_x, :], Proj[idx_y, :], color=X_label, colormap=(:batlow, 0.6))
    col += 1
    if col > 2
      col = 1
      row += 1
    end
  end
  return fig
end


export plot3D_data, plot_matrix_coords
