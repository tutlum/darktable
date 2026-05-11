/*
    This file is part of darktable,
    Copyright (C) 2024 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "bauhaus/bauhaus.h"
#include "common/math.h"
#include "common/opencl.h"
#include "control/control.h"
#include "develop/blend.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "dtgtk/resetlabel.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"
#include <gtk/gtk.h>
#include <inttypes.h>

DT_MODULE_INTROSPECTION(1, dt_iop_deflicker_params_t)

typedef enum dt_iop_deflicker_orientation_t
{
  ORIENTATION_HORIZONTAL = 0, // $DESCRIPTION: "horizontal bands"
  ORIENTATION_VERTICAL = 1    // $DESCRIPTION: "vertical bands"
} dt_iop_deflicker_orientation_t;

typedef struct dt_iop_deflicker_params_t
{
  dt_iop_deflicker_orientation_t orientation; // $DEFAULT: ORIENTATION_HORIZONTAL
  float offset;           // $MIN: 0.0 $MAX: 10000.0 $DEFAULT: 50.0 $DESCRIPTION: "offset" Position of first line in pixels
  float period;           // $MIN: 1.0 $MAX: 2000.0 $DEFAULT: 100.0 $DESCRIPTION: "period" Distance between lines in pixels
  float width;            // $MIN: 1.0 $MAX: 1500.0 $DEFAULT: 10.0 $DESCRIPTION: "band width" Width of underexposed band in pixels
  float feather_start;    // $MIN: 0.0 $MAX: 500.0 $DEFAULT: 10.0 $DESCRIPTION: "feather start" Feathering on leading edge in pixels
  float feather_end;      // $MIN: 0.0 $MAX: 500.0 $DEFAULT: 10.0 $DESCRIPTION: "feather end" Feathering on trailing edge in pixels
  float brightness;       // $MIN: -2.0 $MAX: 4.0 $DEFAULT: 0.5 $DESCRIPTION: "brightness" Exposure compensation in EV
  gboolean invert;        // $DEFAULT: FALSE $DESCRIPTION: "hide guides" Hide the guides to better see brightnesadjustment or finetuning
  gboolean unbound;       // $DEFAULT: TRUE $DESCRIPTION: "unbound" Allow values beyond [0,1]
} dt_iop_deflicker_params_t;

typedef struct dt_iop_deflicker_gui_data_t
{
  GtkWidget *orientation;
  GtkWidget *offset;
  GtkWidget *period;
  GtkWidget *width;
  GtkWidget *feather_start;
  GtkWidget *feather_end;
  GtkWidget *brightness;
  GtkWidget *invert;
  GtkLabel *image_info;   // Label to show current image dimensions
  int dragging;           // 0=none, 1=offset, 2=period
  float click_pos;        // Position of first click for measurement
} dt_iop_deflicker_gui_data_t;

typedef struct dt_iop_deflicker_data_t
{
  int orientation;
  float offset;
  float period;
  float width;
  float feather_start;
  float feather_end;
  float brightness;
  gboolean invert;
  gboolean unbound;
} dt_iop_deflicker_data_t;

typedef struct dt_iop_deflicker_global_data_t
{
  int kernel_deflicker;
} dt_iop_deflicker_global_data_t;


const char *name()
{
  return _("deflicker");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("remove flickering bands from rolling shutter or artificial lighting"),
                                      _("corrective"),
                                      _("linear, RGB, scene-referred"),
                                      _("linear, RGB"),
                                      _("linear, RGB, scene-referred"));
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_ALLOW_TILING;
}

int default_group()
{
  return IOP_GROUP_CORRECT;
}

int operation_tags()
{
  return IOP_TAG_DISTORT;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

// Calculate the correction weight for a given position (all values in pixels)
// offset is the CENTER of the band, width and feathering extend from center
static inline float calculate_weight(const float pos, 
                                     const float offset,
                                     const float period,
                                     const float width,
                                     const float feather_start,
                                     const float feather_end,
                                     const gboolean linear)
{
  if(period <= 0.0f || width <= 0.0f) return 0.0f;
  
  // Calculate position within the repeating pattern (centered on offset)
  float phase = fmodf(pos - offset + 1000.0f * period, period);
  if(phase < 0.0f) phase += period;
  
  // Band structure: [feather_start][width/2][CENTER][width/2][feather_end]
  // Distance from band center
  float dist_from_center = phase;
  if(dist_from_center < 0.0f) dist_from_center += period;
  if(dist_from_center > period * 0.5f) dist_from_center -= period;
  
  const float half_width = width * 0.5f;
  const float abs_dist = fabsf(dist_from_center);
  
  // Determine which side of center we're on
  const gboolean before_center = (dist_from_center < 0.0f);
  const float feather = before_center ? feather_start : feather_end;
  
  float weight = 0.0f;
  
  if(abs_dist <= half_width)
  {
    // Inside the full correction zone
    weight = 1.0f;
  }
  else if(abs_dist <= half_width + feather && feather > 0.0f)
  {
    // In feathering zone - use cosine falloff
    const float feather_pos = abs_dist - half_width;
    const float t = feather_pos / feather;  // 0 to 1
    if (linear) {
      weight = t;
    } else {
      weight = 0.5f * (1.0f + cosf(t * M_PI));  // Smooth cosine falloff
    }
    
  }
  
  return CLAMP(weight, 0.0f, 1.0f);
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  if(!dt_iop_have_required_input_format(4 /*we need full-color pixels*/,
                                        self, piece->colors,
                                        ivoid, ovoid, roi_in, roi_out))
    return;

  const dt_iop_deflicker_data_t *data = piece->data;
  const gboolean unbound = data->unbound;
  
  // All parameters are already in pixels
  const float offset_px = data->offset;
  const float period_px = data->period;
  const float width_px = data->width;
  const float feather_start_px = data->feather_start;
  const float feather_end_px = data->feather_end;
  const gboolean linear = true;
  
  // Convert brightness from EV to linear multiplier
  const float brightness_mult = powf(2.0f, data->brightness);

  DT_OMP_FOR()
  for(int j = 0; j < roi_out->height; j++)
  {
    const size_t k = (size_t)4 * roi_out->width * j;
    const float *in = (const float *)ivoid + k;
    float *out = (float *)ovoid + k;
    
    for(int i = 0; i < roi_out->width; i++)
    {
      // Calculate position in full image coordinates (pixels)
      const float global_i = (i + roi_out->x) / roi_out->scale;
      const float global_j = (j + roi_out->y) / roi_out->scale;
      
      // Determine position along band direction
      const float pos = (data->orientation == ORIENTATION_HORIZONTAL) 
                        ? global_j : global_i;
      
      // Calculate correction weight
      float weight = calculate_weight(pos, offset_px, period_px, width_px,
                                     feather_start_px, feather_end_px, linear);
      
      // Apply correction
      dt_aligned_pixel_t col;
      copy_pixel(col, in + 4*i);
      
      if(weight > 0.0f)
      {
        const float correction = 1.0f + weight * (brightness_mult - 1.0f);
        
        for_each_channel(c)
        {
          col[c] = col[c] * correction;
          col[c] = unbound ? col[c] : CLAMP(col[c], 0.0f, 1.0f);
        }
      }
      
      copy_pixel_nontemporal(out + 4*i, col);
    }
  }
}

#ifdef HAVE_OPENCL
int process_cl(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in, cl_mem dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  dt_iop_deflicker_data_t *data = piece->data;
  dt_iop_deflicker_global_data_t *gd = self->global_data;

  const int devid = piece->pipe->devid;
  const int width = roi_out->width;
  const int height = roi_out->height;

  // Parameters are already in pixels
  const float offset_px = data->offset;
  const float period_px = data->period;
  const float width_px = data->width;
  const float feather_start_px = data->feather_start;
  const float feather_end_px = data->feather_end;
  const float brightness_mult = powf(2.0f, data->brightness);
  const int linear = 1;
  
  const int orientation = data->orientation;
  const int invert = data->invert;
  const int unbound = data->unbound;
  const float scale = roi_out->scale;
  const int roi_x = roi_out->x;
  const int roi_y = roi_out->y;

  return dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_deflicker, width, height,
    CLARG(dev_in), CLARG(dev_out), CLARG(width), CLARG(height),
    CLARG(orientation), CLARG(offset_px), CLARG(period_px), CLARG(width_px),
    CLARG(feather_start_px), CLARG(feather_end_px), CLARG(brightness_mult), CLARG(linear),
    CLARG(invert), CLARG(unbound), CLARG(scale), CLARG(roi_x), CLARG(roi_y));
}
#endif

void init_global(dt_iop_module_so_t *self)
{
  const int program = 8; // extended.cl from programs.conf
  dt_iop_deflicker_global_data_t *gd = malloc(sizeof(dt_iop_deflicker_global_data_t));
  self->data = gd;
  gd->kernel_deflicker = dt_opencl_create_kernel(program, "deflicker");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_deflicker_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_deflicker);
  free(self->data);
  self->data = NULL;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_deflicker_params_t *p = (dt_iop_deflicker_params_t *)p1;
  dt_iop_deflicker_data_t *d = piece->data;
  
  d->orientation = p->orientation;
  d->offset = p->offset;
  d->period = p->period;
  d->width = p->width;
  d->feather_start = p->feather_start;
  d->feather_end = p->feather_end;
  d->brightness = p->brightness;
  d->invert = p->invert;
  d->unbound = p->unbound;
}

void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_deflicker_data_t));
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

// Interactive overlay drawing
void gui_post_expose(dt_iop_module_t *self,
                     cairo_t *cr,
                     const float wd,
                     const float ht,
                     const float pzx,
                     const float pzy,
                     const float zoom_scale)
{
  dt_iop_deflicker_params_t *p = self->params;
  const gboolean invert = p->invert;
  if (invert) return; // no guides if invert is checked
  
  // Get the processing pipeline dimensions at this module
  dt_dev_pixelpipe_iop_t *piece = dt_dev_distort_get_iop_pipe(self->dev, self->dev->preview_pipe, self);
  if(!piece) return;
    
  const gboolean horizontal_in_preview = (p->orientation == ORIENTATION_HORIZONTAL);
    
  // Scale from pipeline pixels to preview display pixels
  const float scale = 1 / piece->pipe->iscale;
  const float offset_display = p->offset * scale;
  const float period_display = p->period * scale;
  const float width_display = p->width * scale;
  const float feather_start_display = p->feather_start * scale;
  const float feather_end_display = p->feather_end * scale;
  
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0) / zoom_scale);
  
  // Draw up to 60 bands visible on screen
  const float display_dimension = horizontal_in_preview ? ht : wd;
  const int max_bands = (period_display > 0.0f) ? (int)((display_dimension / period_display) + 2) : 0;
  
  for(int band = - MIN(max_bands, 30); band < MIN(max_bands, 30); band++)
  {
    // Band center position
    const float band_center = offset_display + band * period_display;
    
    // Band extends from center
    const float half_width = width_display * 0.5f;
    const float band_inner_start = band_center - half_width;
    const float band_inner_end = band_center + half_width;
    // const float avg_feather = (feather_start_display + feather_end_display) * 0.5f;
    const float band_outer_start = band_inner_start - feather_start_display;
    const float band_outer_end = band_inner_end + feather_end_display;
    
    if(band_outer_start > display_dimension) break;
    if(band_outer_end < 0) continue;
    
    // Draw center line (solid, bright) - this is where offset points
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0) / zoom_scale);
    dt_draw_set_color_overlay(cr, TRUE, 0.3);
    if(horizontal_in_preview)
    {
      cairo_move_to(cr, 0, band_center);
      cairo_line_to(cr, wd, band_center);
    }
    else
    {
      cairo_move_to(cr, band_center, 0);
      cairo_line_to(cr, band_center, ht);
    }
    cairo_stroke(cr);
    
    
    // Draw inner band boundaries (solid, medium) - width extends from center
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0) / zoom_scale);
    double dashes = 10.0 / zoom_scale;
    cairo_set_dash(cr, &dashes, 1, 0);
    dt_draw_set_color_overlay(cr, TRUE, 0.7);
    if(horizontal_in_preview)
    {
      cairo_move_to(cr, 0, band_inner_start);
      cairo_line_to(cr, wd, band_inner_start);
      cairo_move_to(cr, 0, band_inner_end);
      cairo_line_to(cr, wd, band_inner_end);
    }
    else
    {
      cairo_move_to(cr, band_inner_start, 0);
      cairo_line_to(cr, band_inner_start, ht);
      cairo_move_to(cr, band_inner_end, 0);
      cairo_line_to(cr, band_inner_end, ht);
    }
    cairo_stroke(cr);
    
    // Draw feather boundaries (dashed)
    dt_draw_set_color_overlay(cr, TRUE, 0.4);
    double dashess[] = {10.0 / zoom_scale, 20.0 / zoom_scale}; // dashes = 20.0 / zoom_scale; // 
    cairo_set_dash(cr, dashess, 2, 0);
    
    if(horizontal_in_preview)
    {
      cairo_move_to(cr, 0, band_outer_start);
      cairo_line_to(cr, wd, band_outer_start);
      cairo_move_to(cr, 0, band_outer_end);
      cairo_line_to(cr, wd, band_outer_end);
    }
    else
    {
      cairo_move_to(cr, band_outer_start, 0);
      cairo_line_to(cr, band_outer_start, ht);
      cairo_move_to(cr, band_outer_end, 0);
      cairo_line_to(cr, band_outer_end, ht);
    }
    cairo_stroke(cr);
    cairo_set_dash(cr, NULL, 0, 0);
  }
}

int mouse_moved(dt_iop_module_t *self,
                const float pzx,
                const float pzy,
                const double pressure,
                const int which,
                const float zoom_scale)
{
  dt_iop_deflicker_gui_data_t *g = self->gui_data;
  dt_iop_deflicker_params_t *p = self->params;
  
  // Get the processing pipeline dimensions
  dt_dev_pixelpipe_iop_t *piece = dt_dev_distort_get_iop_pipe(self->dev, self->dev->preview_pipe, self);
  if(!piece) return 0;
  
  const dt_iop_roi_t *buf_in = &piece->buf_in;
  
  const gboolean horizontal = (p->orientation == ORIENTATION_HORIZONTAL);
  
  // Convert normalized preview coordinates to processing pixels
  // pzx, pzy are in [0,1] normalized to the PREVIEW window
  // We need to map to processing coordinates
  const float scale = piece->pipe->iscale;
  const float pos_full = horizontal ? (pzy * buf_in->height * scale) : (pzx * buf_in->width * scale);
  
  if(darktable.control->button_down && darktable.control->button_down_which == GDK_BUTTON_PRIMARY)
  {
    if(g->dragging == 1)
    {
      // Dragging to set offset
      dt_bauhaus_slider_set(g->offset, pos_full);
      dt_control_queue_redraw_center();
      return 1;
    }
    else if(g->dragging == 2)
    {
      // Dragging to measure period
      const float distance = fabsf(pos_full - g->click_pos);
      dt_bauhaus_slider_set(g->period, distance);
      dt_control_queue_redraw_center();
      return 1;
    }
  }
  
  return 0;
}

int button_pressed(dt_iop_module_t *self,
                   const float pzx,
                   const float pzy,
                   const double pressure,
                   const int which,
                   const int type,
                   const uint32_t state,
                   const float zoom_scale)
{
  if(which == GDK_BUTTON_PRIMARY)
  {
    dt_iop_deflicker_gui_data_t *g = self->gui_data;
    dt_iop_deflicker_params_t *p = self->params;
    
    // Get processing dimensions
    dt_dev_pixelpipe_iop_t *piece = dt_dev_distort_get_iop_pipe(self->dev, self->dev->preview_pipe, self);
    if(!piece) return 0;
    
    const dt_iop_roi_t *buf_in = &piece->buf_in;
    
    const gboolean horizontal = (p->orientation == ORIENTATION_HORIZONTAL);
    
    // Convert normalized coordinates to processing pixels
    // We need to map to processing coordinates
    const float scale = piece->pipe->iscale;
    const float pos_full = horizontal ? (pzy * buf_in->height * scale) : (pzx * buf_in->width * scale);
  
    if(state & GDK_SHIFT_MASK)
    {
      // Shift+Click: Measure period from this point
      g->dragging = 2;
      g->click_pos = pos_full;
    }
    else
    {
      // Normal click: Set offset to clicked position
      g->dragging = 1;
    }
    dt_bauhaus_slider_set(g->offset, pos_full);
    dt_control_queue_redraw_center();
    return 1;
  }
  return 0;
}

int button_released(dt_iop_module_t *self,
                    const float pzx,
                    const float pzy,
                    const int which,
                    const uint32_t state,
                    const float zoom_scale)
{
  if(which == GDK_BUTTON_PRIMARY)
  {
    dt_iop_deflicker_gui_data_t *g = self->gui_data;
    g->dragging = 0;
    return 1;
  }
  return 0;
}

int scrolled(dt_iop_module_t *self,
             const float pzx,
             const float pzy,
             const int up,
             const uint32_t state)
{
  dt_iop_deflicker_gui_data_t *g = self->gui_data;
  dt_iop_deflicker_params_t *p = self->params;
  
  // Scroll to adjust bandwidth
  if(state == GDK_CONTROL_MASK) // No modifiers
  {
    const float step = up ? 1.0f : -1.0f;
    const float new_width = CLAMP(p->width + step, 1.0f, 1000.0f);
    dt_bauhaus_slider_set(g->width, new_width);
    dt_control_queue_redraw_center();
    return 1;
  } else if(state == GDK_SHIFT_MASK) // No modifiers
  {
    const float step = up ? 1.0f : -1.0f;
    const float new_feather_start = CLAMP(p->feather_start + step, 1.0f, 1000.0f);
    dt_bauhaus_slider_set(g->feather_start, new_feather_start);
    const float new_feather_end = CLAMP(p->feather_end + step, 1.0f, 1000.0f);
    dt_bauhaus_slider_set(g->feather_end, new_feather_end);
    dt_control_queue_redraw_center();
    return 1;
  }
  
  return 0;
}

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_deflicker_gui_data_t *g = self->gui_data;
  
  // Update the info label with current image dimensions
  dt_dev_pixelpipe_iop_t *piece = dt_dev_distort_get_iop_pipe(self->dev, self->dev->preview_pipe, self);
  if(piece && piece->buf_in.width > 0)
  {
    dt_iop_deflicker_params_t *p = self->params;
    const gboolean horizontal = (p->orientation == ORIENTATION_HORIZONTAL);
    const int img_width = piece->buf_in.width;
    const int img_height = piece->buf_in.height;
    const int dimension = horizontal ? img_height : img_width;
    
    gchar *text = g_strdup_printf(_("pipeline size: %d × %d px\nrelevant dimension: %d px"),
                                  img_width, img_height, dimension);
    gtk_label_set_text(g->image_info, text);
    g_free(text);
  }
}

void gui_update(dt_iop_module_t *self)
{
  // Update info label
  gui_changed(self, NULL, NULL);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_deflicker_gui_data_t *g = IOP_GUI_ALLOC(deflicker);
  
  g->dragging = 0;
  g->click_pos = 0.0f;

  g->orientation = dt_bauhaus_combobox_from_params(self, "orientation");
  
  // Add image info label
  g->image_info = GTK_LABEL(gtk_label_new(_("pipeline size: loading...")));
  gtk_label_set_line_wrap(GTK_LABEL(g->image_info), TRUE);
  gtk_label_set_xalign(GTK_LABEL(g->image_info), 0.0);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->image_info), FALSE, FALSE, 0);
  
  gtk_box_pack_start(GTK_BOX(self->widget),
                     dt_ui_section_label_new(C_("section", "band pattern (pixels)")),
                     FALSE, FALSE, 0);
  
  g->offset = dt_bauhaus_slider_from_params(self, "offset");
  dt_bauhaus_slider_set_format(g->offset, " px");
  dt_bauhaus_slider_set_digits(g->offset, 1);
  dt_bauhaus_slider_set_step(g->offset, 1.0);
  gtk_widget_set_tooltip_text(g->offset, 
    _("position of band CENTER in pixels\nclick on dark band center to set"));
  
  g->period = dt_bauhaus_slider_from_params(self, "period");
  dt_bauhaus_slider_set_format(g->period, " px");
  dt_bauhaus_slider_set_digits(g->period, 1);
  dt_bauhaus_slider_set_step(g->period, 1.0);
  gtk_widget_set_tooltip_text(g->period, 
    _("distance between band centers in pixels\nShift+drag from first to last band, set number of bands below"));
  
  g->width = dt_bauhaus_slider_from_params(self, "width");
  dt_bauhaus_slider_set_format(g->width, " px");
  dt_bauhaus_slider_set_digits(g->width, 1);
  dt_bauhaus_slider_set_step(g->width, 1.0);
  gtk_widget_set_tooltip_text(g->width, 
    _("full width of band (extends width/2 on each side of center)\nuse mouse scroll to adjust"));
  
  gtk_box_pack_start(GTK_BOX(self->widget),
                     dt_ui_section_label_new(C_("section", "blending (pixels)")),
                     FALSE, FALSE, 0);
  
  g->feather_start = dt_bauhaus_slider_from_params(self, "feather_start");
  dt_bauhaus_slider_set_format(g->feather_start, " px");
  dt_bauhaus_slider_set_digits(g->feather_start, 1);
  dt_bauhaus_slider_set_step(g->feather_start, 1.0);
  gtk_widget_set_tooltip_text(g->feather_start, 
    _("feathering on leading edge (before center)\ncosine falloff for smooth transition"));
  
  g->feather_end = dt_bauhaus_slider_from_params(self, "feather_end");
  dt_bauhaus_slider_set_format(g->feather_end, " px");
  dt_bauhaus_slider_set_digits(g->feather_end, 1);
  dt_bauhaus_slider_set_step(g->feather_end, 1.0);
  gtk_widget_set_tooltip_text(g->feather_end, 
    _("feathering on trailing edge (after center)\ncosine falloff for smooth transition"));
  
  gtk_box_pack_start(GTK_BOX(self->widget),
                     dt_ui_section_label_new(C_("section", "correction")),
                     FALSE, FALSE, 0);
  
  g->brightness = dt_bauhaus_slider_from_params(self, "brightness");
  dt_bauhaus_slider_set_format(g->brightness, " EV");
  dt_bauhaus_slider_set_digits(g->brightness, 2);
  gtk_widget_set_tooltip_text(g->brightness, 
    _("exposure compensation in EV for underexposed bands"));
  
  g->invert = dt_bauhaus_toggle_from_params(self, "invert");
  gtk_widget_set_tooltip_text(g->invert, 
    _("correct overexposed bands instead of underexposed"));
}

GSList *mouse_actions(dt_iop_module_t *self)
{
  GSList *lm = NULL;
  lm = dt_mouse_action_create_format(lm, DT_MOUSE_ACTION_LEFT, GDK_CONTROL_MASK,
    _("[%s] set offset to clicked position"), self->name());
  lm = dt_mouse_action_create_format(lm, DT_MOUSE_ACTION_LEFT_DRAG, GDK_SHIFT_MASK,
    _("[%s] set offset and measure period between bands"), self->name());
  lm = dt_mouse_action_create_format(lm, DT_MOUSE_ACTION_SCROLL, GDK_CONTROL_MASK,
    _("[%s] adjust band width"), self->name());
  lm = dt_mouse_action_create_format(lm, DT_MOUSE_ACTION_SCROLL, GDK_SHIFT_MASK,
    _("[%s] adjust feather width"), self->name());
  return lm;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on