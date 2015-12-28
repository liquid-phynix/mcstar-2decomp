module iop

    use MPI
    use decomp_2d
    use iso_c_binding

    implicit none
    integer :: ierror

contains

    subroutine get_float_size(s)
        integer (c_int), intent(out) :: s
        s = mytype_bytes
    end subroutine

    subroutine sync_configuration(out_xstart, out_ystart, out_zstart, &
                                  out_xend, out_yend, out_zend,       &
                                  out_xsize, out_ysize, out_zsize)

        integer (c_int), dimension(3), intent(out) :: out_xstart, &
            out_ystart, out_zstart, out_xend, out_yend, out_zend, &
            out_xsize, out_ysize, out_zsize

        out_xstart(:) = xstart(:) - 1
        out_ystart(:) = ystart(:) - 1
        out_zstart(:) = zstart(:) - 1
        out_xend  (:) = xend  (:) - 1
        out_yend  (:) = yend  (:) - 1
        out_zend  (:) = zend  (:) - 1
        out_xsize (:) = xsize (:)
        out_ysize (:) = ysize (:)
        out_zsize (:) = zsize (:)

    end subroutine

    subroutine iop_transpose(ttype, input_ptr, input_sizes, output_ptr, output_sizes)
        ! ttype = (1: x->y) | (2: y->z) | (3: z->y) | (4: y->x)
        integer (c_int), intent(in) :: ttype
        integer (c_int), dimension(3), intent(in) :: input_sizes, output_sizes
        type (c_ptr)                :: input_ptr, output_ptr
        complex(mytype), pointer    :: input(:,:,:), output(:,:,:)

        call c_f_pointer(input_ptr, input, input_sizes)
        call c_f_pointer(output_ptr, output, output_sizes)

        select case (ttype)
        case (1)
            call transpose_x_to_y(input, output)
        case (2)
            call transpose_y_to_z(input, output)
        case (3)
            call transpose_z_to_y(input, output)
        case (4)
            call transpose_y_to_x(input, output)
        case default
            stop "ttype not in [1,4]"
        end select

    end subroutine

end module
