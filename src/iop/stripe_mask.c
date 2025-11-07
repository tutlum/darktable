/*
    This file is part of darktable,
    copyright (c) 2025

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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "bauhaus/bauhaus.h"
#include "common/imagebuf.h"
#include "common/opencl.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "dtgtk/button.h"
#include "dtgtk/resetlabel.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
#include <assert.h>
#include <gtk/gtk.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

DT_MODULE_INTROSPECTION(1, dt_iop_stripe_mask_params_t)

typedef enum dt_iop_stripe_mask_direction_t
{
  STRIPE_VERTICAL = 0,
  STRIPE_HORIZONTAL = 1
} dt_iop_stripe_mask_direction_t;

typedef struct dt_iop_stripe_mask_params_t
{
  float spacing;              // Spacing between stripe centers in pixels
  float width;                // Width of each stripe in pixels
  float feather_start;        // Feathering at the start edge in pixels
  float feather_end;          // Feathering at the end edge in pixels
  float offset;               // Offset of the first stripe in pixels
  float opacity;              // Overall opacity/intensity of the effect
  int direction;              // 0 = vertical, 1 = horizontal
} dt_iop_stripe_mask_params_t;

typedef struct dt_iop_stripe_mask_gui_data_t
{
  GtkWidget *spacing;
  GtkWidget *width;
  GtkWidget *feather_start;
  GtkWidget *feather_end;
  GtkWidget *offset;
  GtkWidget *opacity;
  GtkWidget *direction;
} dt_iop_stripe_mask_gui_data_t;

typedef struct dt_iop_stripe_mask_data_t
{
  float spacing;
  float width;
  float feather_start;
  float feather_end;
  float offset;
  float opacity;
  int direction;
} dt_iop_stripe_mask_data_t;

const char *name()
{
  return _("repeating stripe mask");
}

const char **description(struct dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("create repeating stripe patterns for masking"),
                                      _("corrective"),
                                      _("linear, RGB, scene-referred"),
                                      _("linear, RGB"),
                                      _("linear, RGB, scene-referred"));
}

int default_group()
{
  return IOP_GROUP_EFFECTS | IOP_GROUP_GRADING;
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_ALLOW_TILING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self, 
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_stripe_mask_data_t));
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_stripe_mask_params_t *p = (dt_iop_stripe_mask_params_t *)p1;
  dt_iop_stripe_mask_data_t *d = (dt_iop_stripe_mask_data_t *)piece->data;
  
  d->spacing = p->spacing;
  d->width = p->width;
  d->feather_start = p->feather_start;
  d->feather_end = p->feather_end;
  d->offset = p->offset;
  d->opacity = p->opacity;
  d->direction = p->direction;
}

void init_global(dt_iop_module_so_t *module)
{
  module->data = NULL;
}

void cleanup_global(dt_iop_module_so_t *module)
{
}

// Gaussian function for smooth feathering
static inline float gaussian_weight(float x, float sigma)
{
  if(sigma <= 0.0f) return 1.0f;
  return expf(-(x * x) / (2.0f * sigma * sigma));
}

// Calculate mask value for a given position
static inline float calculate_stripe_mask(float pos, float spacing, float width,
                                          float feather_start, float feather_end,
                                          float offset)
{
  // Adjust position by offset
  pos -= offset;
  
  // Find position within the repeating pattern
  float pattern_pos = fmodf(pos, spacing);
  if(pattern_pos < 0.0f) pattern_pos += spacing;
  
  const float half_width = width * 0.5f;
  const float center = spacing * 0.5f;
  const float dist_from_center = fabsf(pattern_pos - center);
  
  // Check if we're in the stripe region
  if(dist_from_center > half_width + fmaxf(feather_start, feather_end))
  {
    return 0.0f; // Outside stripe and feather region
  }
  
  // Inside the core stripe
  if(dist_from_center <= half_width)
  {
    return 1.0f;
  }
  
  // In feather region
  const float dist_from_edge = dist_from_center - half_width;
  
  // Determine which side we're on
  const float is_start_side = (pattern_pos < center) ? 1.0f : 0.0f;
  const float feather_size = is_start_side * feather_start + (1.0f - is_start_side) * feather_end;
  
  if(feather_size <= 0.0f) return 0.0f;
  
  // Calculate gaussian falloff
  const float sigma = feather_size / 2.5f; // 2.5 gives a smooth falloff
  return gaussian_weight(dist_from_edge, sigma);
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_stripe_mask_data_t *d = (dt_iop_stripe_mask_data_t *)piece->data;
  
  const int ch = piece->colors;
  const float opacity = d->opacity;
  
  // Scale parameters according to ROI scale
  const float scale = roi_in->scale / piece->iscale;
  const float spacing = d->spacing * scale;
  const float width = d->width * scale;
  const float feather_start = d->feather_start * scale;
  const float feather_end = d->feather_end * scale;
  const float offset = d->offset * scale;
  
  const size_t width_out = roi_out->width;
  const size_t height_out = roi_out->height;
  
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(ch, ivoid, ovoid, width_out, height_out, spacing, width, \
                      feather_start, feather_end, offset, opacity, d) \
  schedule(static)
#endif
  for(size_t j = 0; j < height_out; j++)
  {
    const float *in = ((float *)ivoid) + (size_t)ch * j * width_out;
    float *out = ((float *)ovoid) + (size_t)ch * j * width_out;
    
    for(size_t i = 0; i < width_out; i++)
    {
      // Calculate mask value based on direction
      float mask_value;
      if(d->direction == STRIPE_VERTICAL)
      {
        mask_value = calculate_stripe_mask((float)i, spacing, width, 
                                          feather_start, feather_end, offset);
      }
      else // STRIPE_HORIZONTAL
      {
        mask_value = calculate_stripe_mask((float)j, spacing, width,
                                          feather_start, feather_end, offset);
      }
      
      // Apply opacity
      mask_value *= opacity;
      
      // Apply mask to input (lighten dark stripes)
      // This creates a brightening effect on the stripe areas
      const float boost = 1.0f + mask_value * 0.5f; // Adjust multiplier as needed
      
      for(int c = 0; c < 3; c++)
      {
        out[ch * i + c] = in[ch * i + c] * boost;
      }
      
      // Copy alpha channel unchanged
      if(ch == 4)
      {
        out[ch * i + 3] = in[ch * i + 3];
      }
    }
  }
  
  // Note: Darktable's blending system will handle the final compositing
  // based on the blend mode and opacity set in the blending controls
}

void init_presets(dt_iop_module_so_t *self)
{
  // Default preset for typical rolling shutter issues
  dt_iop_stripe_mask_params_t p;
  memset(&p, 0, sizeof(p));
  
  p.spacing = 123.0f;
  p.width = 16.0f;
  p.feather_start = 2.0f;
  p.feather_end = 2.0f;
  p.offset = 0.0f;
  p.opacity = 1.0f;
  p.direction = STRIPE_VERTICAL;
  
  dt_gui_presets_add_generic(_("rolling shutter (vertical)"), self->op,
                            self->version(), &p, sizeof(p), 1, DEVELOP_BLEND_CS_RGB_DISPLAY);
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_stripe_mask_gui_data_t *g = (dt_iop_stripe_mask_gui_data_t *)self->gui_data;
  dt_iop_stripe_mask_params_t *p = (dt_iop_stripe_mask_params_t *)self->params;
  
  dt_bauhaus_combobox_set(g->direction, p->direction);
  dt_bauhaus_slider_set(g->spacing, p->spacing);
  dt_bauhaus_slider_set(g->width, p->width);
  dt_bauhaus_slider_set(g->offset, p->offset);
  dt_bauhaus_slider_set(g->feather_start, p->feather_start);
  dt_bauhaus_slider_set(g->feather_end, p->feather_end);
  dt_bauhaus_slider_set(g->opacity, p->opacity);
}

void gui_reset(dt_iop_module_t *self)
{
  // Reset is handled by darktable's parameter system
}

void init(dt_iop_module_t *module)
{
  dt_iop_default_init(module);
  
  dt_iop_stripe_mask_params_t *d = module->default_params;
  d->spacing = 123.0f;
  d->width = 16.0f;
  d->feather_start = 2.0f;
  d->feather_end = 2.0f;
  d->offset = 0.0f;
  d->opacity = 1.0f;
  d->direction = STRIPE_VERTICAL;
}

void gui_init(dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_stripe_mask_gui_data_t));
  dt_iop_stripe_mask_gui_data_t *g = (dt_iop_stripe_mask_gui_data_t *)self->gui_data;
  
  // Direction selector
  g->direction = dt_bauhaus_combobox_from_widget(self, "direction");
  dt_bauhaus_combobox_add(g->direction, _("vertical"));
  dt_bauhaus_combobox_add(g->direction, _("horizontal"));
  gtk_widget_set_tooltip_text(g->direction, _("direction of the stripes"));
  
  // Spacing slider
  g->spacing = dt_bauhaus_slider_from_widget(self, "spacing");
  dt_bauhaus_slider_set_format(g->spacing, "%.0f px");
  dt_bauhaus_slider_set_range(g->spacing, 10.0, 500.0);
  dt_bauhaus_slider_set_step(g->spacing, 1.0);
  gtk_widget_set_tooltip_text(g->spacing, _("distance between stripe centers"));
  
  // Width slider
  g->width = dt_bauhaus_slider_from_widget(self, "width");
  dt_bauhaus_slider_set_format(g->width, "%.0f px");
  dt_bauhaus_slider_set_range(g->width, 1.0, 100.0);
  dt_bauhaus_slider_set_step(g->width, 1.0);
  gtk_widget_set_tooltip_text(g->width, _("width of each stripe"));
  
  // Offset slider
  g->offset = dt_bauhaus_slider_from_widget(self, "offset");
  dt_bauhaus_slider_set_format(g->offset, "%.0f px");
  dt_bauhaus_slider_set_range(g->offset, -250.0, 250.0);
  dt_bauhaus_slider_set_step(g->offset, 1.0);
  gtk_widget_set_tooltip_text(g->offset, _("shift stripe pattern position"));
  
  // Feather start slider
  g->feather_start = dt_bauhaus_slider_from_widget(self, "feather_start");
  dt_bauhaus_slider_set_format(g->feather_start, "%.1f px");
  dt_bauhaus_slider_set_range(g->feather_start, 0.0, 20.0);
  dt_bauhaus_slider_set_step(g->feather_start, 0.1);
  gtk_widget_set_tooltip_text(g->feather_start, _("feathering at leading edge"));
  
  // Feather end slider
  g->feather_end = dt_bauhaus_slider_from_widget(self, "feather_end");
  dt_bauhaus_slider_set_format(g->feather_end, "%.1f px");
  dt_bauhaus_slider_set_range(g->feather_end, 0.0, 20.0);
  dt_bauhaus_slider_set_step(g->feather_end, 0.1);
  gtk_widget_set_tooltip_text(g->feather_end, _("feathering at trailing edge"));
  
  // Opacity slider
  g->opacity = dt_bauhaus_slider_from_widget(self, "opacity");
  dt_bauhaus_slider_set_format(g->opacity, "%.0f%%");
  dt_bauhaus_slider_set_factor(g->opacity, 100.0);
  dt_bauhaus_slider_set_range(g->opacity, 0.0, 1.0);
  dt_bauhaus_slider_set_step(g->opacity, 0.01);
  gtk_widget_set_tooltip_text(g->opacity, _("overall effect intensity"));
}

void gui_cleanup(dt_iop_module_t *self)
{
  g_free(self->gui_data);
  self->gui_data = NULL;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on